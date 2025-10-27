LDM = True
IMAGE_NET = True
DROPOUT = 0.1
EMA_DECAY = 0.999
NUM_TIME_STEPS = 500
USING_LPIPS = False
IMAGENET_ID_TO_NAME = {
    19: 'airliner',
    363: 'goldfinch',
    829: 'sports car',
    872: 'tabby',
    353: 'gazelle',
    84: 'beagle',
    816: 'sorrel',
    531: 'macaque',
    233: 'container ship',
    915: 'trailer truck',
    1000: 'uncond'
}

IMAGENET_ID_TO_NAME_WITHOUT_UNCOND = {
    19: 'airliner',
    363: 'goldfinch',
    829: 'sports car',
    872: 'tabby',
    353: 'gazelle',
    84: 'beagle',
    816: 'sorrel',
    531: 'macaque',
    233: 'container ship',
    915: 'trailer truck'
}

NUM_TIME_STEPS_DPM = 100

if IMAGE_NET:
    UNCOND_LABEL = 'uncond'
    UNCOND_ID = 1000

NUM_IMAGES_PER_FOLDER_FOR_SPECTRUM = 10

LOW_T_REWARDING_WARMUP_EPOCHS = 20
LOW_T_REWARDING_WARMUP_WEIGHT = 0.01
USING_LOW_T_REWARDING = True