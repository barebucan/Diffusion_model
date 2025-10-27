from utils import create_imagenet_classes

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
all_classes = create_imagenet_classes()
for i in IMAGENET_ID_TO_NAME.values():
    print(all_classes.index(i), i)
