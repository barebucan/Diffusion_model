import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch


def _ensure_clip_available() -> None:
    try:
        import clip  # noqa: F401
    except Exception:
        print("[error] The 'clip' package is not installed. Install via: pip install git+https://github.com/openai/CLIP.git")
        raise


def _default_paths() -> Tuple[str, str]:
    base = os.path.join('artifacts', 'clip', 'clip_text_emb_stl10_vitb32')
    return base + '.pt', base + '.npz'


def _load_saved_embeddings(path: str) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embeddings file not found: {path}")

    if path.endswith('.pt'):
        obj = torch.load(path, map_location='cpu')
        # Normalize structure
        embeddings = obj['embeddings']
        embeddings_norm = obj.get('embeddings_norm')
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.tensor(embeddings)
        if embeddings_norm is None:
            embeddings_norm = torch.nn.functional.normalize(embeddings.float(), dim=-1)
        elif isinstance(embeddings_norm, np.ndarray):
            embeddings_norm = torch.tensor(embeddings_norm)

        saved = {
            'model_name': obj['model_name'],
            'context_dim': int(obj['context_dim']),
            'class_names': list(obj['class_names']),
            'prompts': obj['prompts'],
            'embeddings': embeddings.float(),
            'embeddings_norm': embeddings_norm.float(),
            'index_mapping': obj.get('index_mapping'),
            'source_path': path,
        }
        return saved

    if path.endswith('.npz'):
        npz = np.load(path, allow_pickle=True)
        class_names = list(npz['class_names'].tolist())
        prompts = {
            'class_prompts': list(npz['class_prompts'].tolist()),
            'uncond_prompt': str(npz['uncond_prompt'].tolist()[0]) if npz['uncond_prompt'].ndim > 0 else str(npz['uncond_prompt'].item()),
        }
        embeddings = torch.tensor(npz['embeddings']).float()
        embeddings_norm = torch.tensor(npz['embeddings_norm']).float()
        saved = {
            'model_name': str(npz['model_name'].tolist()),
            'context_dim': int(npz['context_dim'].tolist()),
            'class_names': class_names,
            'prompts': prompts,
            'embeddings': embeddings,
            'embeddings_norm': embeddings_norm,
            'index_mapping': None,
            'source_path': path,
        }
        return saved

    raise ValueError(f"Unsupported file type: {path}")


@torch.no_grad()
def _recompute_clip_embeddings(model_name: str, class_prompts: List[str], uncond_prompt: str, device: str) -> Dict[str, torch.Tensor]:
    _ensure_clip_available()
    import clip

    model, _ = clip.load(model_name, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    tokens = clip.tokenize(class_prompts).to(device)
    tokens_uncond = clip.tokenize([uncond_prompt]).to(device)

    text_emb = model.encode_text(tokens)
    text_emb_uncond = model.encode_text(tokens_uncond)

    text_emb_norm = torch.nn.functional.normalize(text_emb, dim=-1)
    text_emb_uncond_norm = torch.nn.functional.normalize(text_emb_uncond, dim=-1)

    emb = torch.cat([text_emb, text_emb_uncond], dim=0).float().cpu()
    emb_norm = torch.cat([text_emb_norm, text_emb_uncond_norm], dim=0).float().cpu()
    return {
        'embeddings': emb,
        'embeddings_norm': emb_norm,
    }


def _cosine_similarity_matrix(a_norm: torch.Tensor, b_norm: torch.Tensor) -> torch.Tensor:
    return a_norm @ b_norm.t()


def verify_embeddings(path: str, device: str = None, model_name_override: str = None, topk: int = 3, show_matrix: bool = False) -> int:
    saved = _load_saved_embeddings(path)

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = model_name_override or saved['model_name']

    class_prompts: List[str] = saved['prompts']['class_prompts']
    uncond_prompt: str = saved['prompts']['uncond_prompt']

    fresh = _recompute_clip_embeddings(model_name=model_name, class_prompts=class_prompts, uncond_prompt=uncond_prompt, device=device)

    saved_norm = saved['embeddings_norm'].float().cpu()
    fresh_norm = fresh['embeddings_norm'].float().cpu()

    if saved_norm.ndim != 2 or fresh_norm.ndim != 2:
        print('[error] Embeddings must be rank-2 tensors [N, D].')
        return 2

    if saved_norm.shape != fresh_norm.shape:
        print(f"[error] Shape mismatch. saved={tuple(saved_norm.shape)} fresh={tuple(fresh_norm.shape)}")
        return 2

    n, d = saved_norm.shape
    if saved['context_dim'] != d:
        print(f"[warn] context_dim in file ({saved['context_dim']}) != embedding dim ({d}).")

    sim = _cosine_similarity_matrix(saved_norm, fresh_norm)

    labels = saved['class_names'] + ['uncond']

    diag = sim.diag()
    diag_min = float(diag[:-1].min().item())  # exclude uncond for strictness
    diag_mean = float(diag[:-1].mean().item())
    diag_uncond = float(diag[-1].item())

    print('\n=== CLIP Embedding Verification ===')
    print(f"Source: {saved['source_path']}")
    print(f"Model:  {model_name}")
    print(f"Device: {device}")
    print(f"Shape:  {n} x {d}")

    print('\nPer-class self cosine (saved vs recomputed):')
    for i, name in enumerate(labels):
        print(f"  [{i:2d}] {name:8s} -> cos={float(diag[i].item()):.6f}")

    print('\nTop-{:d} nearest recomputed prompts for each saved row:'.format(topk))
    topk_vals, topk_idx = torch.topk(sim, k=min(topk, n), dim=1)
    for i, name in enumerate(labels):
        idxs = topk_idx[i].tolist()
        vals = topk_vals[i].tolist()
        pairs = ', '.join([f"{labels[j]}({v:.3f})" for j, v in zip(idxs, vals)])
        marker = '' if idxs[0] == i else '  <-- mismatch'
        print(f"  [{i:2d}] {name:8s}: {pairs}{marker}")

    if show_matrix:
        print('\nCosine similarity matrix (saved x recomputed):')
        # bounded printing
        with np.printoptions(precision=3, suppress=True):
            print(sim.numpy())

    # Basic health checks
    ok = True
    # Classes should closely match on the diagonal. CLIP is deterministic; allow small numeric drift.
    if diag_min < 0.999:
        print(f"[warn] Lowest class self-cosine is {diag_min:.6f} (< 0.999). Check prompts/model version.")
        ok = False
    # Unconditional can be looser; warn if surprisingly low
    if diag_uncond < 0.995:
        print(f"[warn] Unconditional self-cosine is {diag_uncond:.6f} (< 0.995).")

    # Validate normalization vs raw
    re_norm = torch.nn.functional.normalize(saved['embeddings'].float(), dim=-1)
    diff = torch.max(torch.abs(re_norm - saved_norm)).item()
    if diff > 1e-5:
        print(f"[warn] embeddings_norm differs from normalized(embeddings) by max {diff:.2e} > 1e-5")

    print('\nSummary:')
    print(f"  Min class self-cosine: {diag_min:.6f}")
    print(f"  Mean class self-cosine: {diag_mean:.6f}")
    print(f"  Uncond self-cosine:     {diag_uncond:.6f}")
    print(f"  Verification:           {'PASS' if ok else 'CHECK' }")

    return 0 if ok else 1


def main():
    pt_default, npz_default = _default_paths()

    parser = argparse.ArgumentParser(description='Verify saved CLIP text embeddings against fresh CLIP encodings.')
    parser.add_argument('--path', type=str, default=pt_default if os.path.exists(pt_default) else (npz_default if os.path.exists(npz_default) else pt_default), help='Path to saved embeddings (.pt or .npz)')
    parser.add_argument('--device', type=str, default=None, help="cuda, cpu, or mps (auto if omitted)")
    parser.add_argument('--model', type=str, default=None, help="Override CLIP model name (default: use saved model)")
    parser.add_argument('--topk', type=int, default=3, help='Top-k neighbors to display per row')
    parser.add_argument('--show-matrix', action='store_true', help='Print full cosine similarity matrix')

    args = parser.parse_args()
    try:
        code = verify_embeddings(path=args.path, device=args.device, model_name_override=args.model, topk=args.topk, show_matrix=args.show_matrix)
    except Exception as e:
        print(f"[error] Verification failed: {e}")
        code = 2
    sys.exit(code)


if __name__ == '__main__':
    main()


