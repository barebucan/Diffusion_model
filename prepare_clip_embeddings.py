import os
import sys
import argparse
import json
from typing import List, Dict, Optional, Tuple

import torch


STL10_CLASSES: List[str] = [
    'airplane', 'bird', 'car', 'cat', 'deer',
    'dog', 'horse', 'monkey', 'ship', 'truck'
]


def _ensure_clip_available():
    try:
        import clip  # noqa: F401
    except Exception as e:
        print("[error] The 'clip' package is not installed. Install via: pip install git+https://github.com/openai/CLIP.git")
        raise


def _sanitize_identifier(name: str) -> str:
    s = name.strip().lower().replace(' ', '_').replace('-', '_').replace('/', '')
    s = ''.join(ch for ch in s if (ch.isalnum() or ch == '_'))
    return s


def _canonicalize_model_name(model_name: str) -> str:
    # e.g., "ViT-B/32" -> "vitb32"
    s = model_name.strip().lower().replace('vit-', 'vit').replace('/', '')
    s = s.replace(' ', '')
    return s


def _load_classes_from_file(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Classes file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.txt', '.list'):
        with open(path, 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f if line.strip()]
        return classes
    if ext == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Support either a list of strings, or mapping {id: [wnid, name]}
        if isinstance(data, list):
            return [str(x) for x in data]
        if isinstance(data, dict):
            # torchvision imagenet_class_index.json style
            try:
                pairs = [data[str(i)][1] for i in range(len(data))]
                return [str(x) for x in pairs]
            except Exception:
                return [str(v) for v in data.values()]
    raise ValueError(f"Unsupported classes file format: {path}")


def build_prompts(classes: List[str], template: str = "a photo of a {}", uncond_prompt: str = "") -> Dict[str, List[str]]:
    prompts = [template.format(c) for c in classes]
    return {
        'class_prompts': prompts,
        'uncond_prompt': uncond_prompt
    }


@torch.no_grad()
def encode_prompts_with_clip(classes: List[str], model_name: str = 'ViT-B/32', device: Optional[str] = None,
                              template: str = "a photo of a {}", uncond_prompt: str = "") -> Dict:
    _ensure_clip_available()
    import clip

    if len(classes) == 0:
        raise ValueError("Classes list is empty.")

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model, _ = clip.load(model_name, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    prompts = build_prompts(classes, template=template, uncond_prompt=uncond_prompt)
    class_prompts: List[str] = prompts['class_prompts']

    # Tokenize
    tokens = clip.tokenize(class_prompts).to(device)
    tokens_uncond = clip.tokenize([prompts['uncond_prompt']]).to(device)

    # Encode text
    text_emb = model.encode_text(tokens)              # [N, D]
    text_emb_uncond = model.encode_text(tokens_uncond)  # [1, D]

    # Normalize (common for CLIP usage)
    text_emb_norm = torch.nn.functional.normalize(text_emb, dim=-1)
    text_emb_uncond_norm = torch.nn.functional.normalize(text_emb_uncond, dim=-1)

    # Stack class + uncond to match index convention (0..N-1 classes, N=uncond)
    emb = torch.cat([text_emb, text_emb_uncond], dim=0).cpu()
    emb_norm = torch.cat([text_emb_norm, text_emb_uncond_norm], dim=0).cpu()

    index_mapping = {i: c for i, c in enumerate(classes)}
    index_mapping[len(classes)] = 'uncond'

    out = {
        'model_name': model_name,
        'context_dim': int(emb.shape[1]),
        'class_names': classes,
        'prompts': {'class_prompts': class_prompts, 'uncond_prompt': prompts['uncond_prompt']},
        'embeddings': emb,               # raw CLIP text features
        'embeddings_norm': emb_norm,     # L2-normalized features
        'index_mapping': index_mapping,
    }
    return out


def _default_paths_for(dataset: Optional[str], model_name: str) -> Tuple[str, str, str]:
    # Returns (out_dir, base_name, flat_dir) where flat_dir keeps backward-compat for STL10
    model_tag = _canonicalize_model_name(model_name)
    if dataset:
        ds_tag = _sanitize_identifier(dataset)
        out_dir = os.path.join('artifacts', 'clip', ds_tag)
        base_name = f'clip_text_emb_{ds_tag}_{model_tag}'
    else:
        out_dir = os.path.join('artifacts', 'clip')
        base_name = f'clip_text_emb_custom_{model_tag}'
    flat_dir = os.path.join('artifacts', 'clip')
    return out_dir, base_name, flat_dir


def save_embeddings(obj: Dict, dataset: Optional[str], out_dir: Optional[str] = None, base_name: Optional[str] = None,
                    keep_stl10_flat_compat: bool = True) -> Tuple[str, str]:
    # Derive defaults
    derived_out_dir, derived_base_name, flat_dir = _default_paths_for(dataset, obj['model_name'])
    out_dir = out_dir or derived_out_dir
    base_name = base_name or derived_base_name

    # Ensure directory
    os.makedirs(out_dir, exist_ok=True)
    torch_path = os.path.join(out_dir, base_name + '.pt')
    npz_path = os.path.join(out_dir, base_name + '.npz')

    # Enrich metadata before saving
    obj = dict(obj)
    obj['dataset'] = dataset
    obj['num_classes'] = len(obj.get('class_names', []))
    obj['filename_base'] = base_name

    # Save torch
    torch.save(obj, torch_path)

    # Save numpy copy (without tensors that aren't arrays)
    import numpy as np
    np.savez(
        npz_path,
        model_name=obj['model_name'],
        context_dim=obj['context_dim'],
        class_names=np.array(obj['class_names'], dtype=object),
        class_prompts=np.array(obj['prompts']['class_prompts'], dtype=object),
        uncond_prompt=np.array([obj['prompts']['uncond_prompt']], dtype=object),
        embeddings=obj['embeddings'].numpy(),
        embeddings_norm=obj['embeddings_norm'].numpy(),
        dataset=np.array([dataset or ''], dtype=object),
        num_classes=int(obj['num_classes']),
        filename_base=np.array([base_name], dtype=object),
    )

    # Backward compatibility for existing STL10 path used elsewhere
    if keep_stl10_flat_compat and (dataset is not None) and (_sanitize_identifier(dataset) == 'stl10'):
        compat_name = f'clip_text_emb_stl10_{_canonicalize_model_name(obj["model_name"])}'
        compat_torch = os.path.join(flat_dir, compat_name + '.pt')
        compat_npz = os.path.join(flat_dir, compat_name + '.npz')
        try:
            if compat_torch != torch_path:
                torch.save(obj, compat_torch)
            if compat_npz != npz_path:
                import numpy as np  # noqa: F401
                # Reuse already written npz by copying bytes
                from shutil import copyfile
                copyfile(npz_path, compat_npz)
        except Exception as e:
            print(f"[warn] Failed to write STL10 flat-compat copies: {e}")

    return torch_path, npz_path


def _resolve_classes(dataset: Optional[str], classes_arg: Optional[str], classes_file: Optional[str]) -> Tuple[List[str], str]:
    # Returns (classes, dataset_resolved)
    if classes_file:
        classes = _load_classes_from_file(classes_file)
        return classes, (dataset or os.path.splitext(os.path.basename(classes_file))[0])
    if classes_arg:
        classes = [c.strip() for c in classes_arg.split(',') if c.strip()]
        if len(classes) == 0:
            raise ValueError("--classes provided but empty after parsing")
        return classes, (dataset or 'custom')
    # Fallback presets
    if dataset and _sanitize_identifier(dataset) == 'stl10':
        return list(STL10_CLASSES), 'stl10'
    # No classes provided; error
    raise ValueError("No classes specified. Provide --classes or --classes-file, or set --dataset stl10 for defaults.")


def main():
    parser = argparse.ArgumentParser(description='Prepare CLIP text embeddings for arbitrary class lists (dataset-aware).')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset tag (e.g., stl10, imagenet1k). Used for metadata and output paths.')
    parser.add_argument('--classes', type=str, default=None, help='Comma-separated class names.')
    parser.add_argument('--classes-file', type=str, default=None, help='Path to classes file (.txt lines or imagenet_class_index.json).')
    parser.add_argument('--model', type=str, default='ViT-B/32', help='CLIP model name (e.g., ViT-B/32, ViT-L/14).')
    parser.add_argument('--template', type=str, default='a photo of a {}', help='Prompt template. Must contain {} placeholder.')
    parser.add_argument('--uncond-prompt', type=str, default='', help='Unconditional prompt text.')
    parser.add_argument('--device', type=str, default=None, help='cuda, cpu, or mps (auto if omitted).')
    parser.add_argument('--out-dir', type=str, default=None, help='Output directory (default: artifacts/clip/{dataset}).')
    parser.add_argument('--save-basename', type=str, default=None, help='Override base file name without extension.')

    args = parser.parse_args()

    try:
        classes, dataset = _resolve_classes(args.dataset, args.classes, args.classes_file)
        obj = encode_prompts_with_clip(
            classes=classes,
            model_name=args.model,
            device=args.device,
            template=args.template,
            uncond_prompt=args.uncond_prompt,
        )
    except Exception as e:
        print(f"[error] Failed to encode prompts with CLIP: {e}")
        sys.exit(1)

    try:
        torch_path, npz_path = save_embeddings(obj, dataset=dataset, out_dir=args.out_dir, base_name=args.save_basename)
    except Exception as e:
        print(f"[error] Failed to save embeddings: {e}")
        sys.exit(1)

    print("Saved CLIP text embeddings:")
    print(f"  Dataset:     {dataset}")
    print(f"  Num classes: {len(obj['class_names'])}")
    print(f"  Model:       {obj['model_name']}")
    print(f"  Context dim: {obj['context_dim']}")
    print(f"  Torch:       {torch_path}")
    print(f"  NPZ:         {npz_path}")


if __name__ == '__main__':
    main()


