import torch
from torchvision.transforms.functional import InterpolationMode
from mobius import MobiusTransform_Improved
def get_module(use_v2):
    # We need a protected import to avoid the V2 warning in case just V1 is used
    if use_v2:
        import torchvision.transforms.v2

        return torchvision.transforms.v2
    else:
        import torchvision.transforms

        return torchvision.transforms


class ClassificationPresetTrain:
    # Note: this transform assumes that the input to forward() are always PIL
    # images, regardless of the backend parameter. We may change that in the
    # future though, if we change the output type from the dataset.
    def __init__(
        self,
        *,
        crop_size,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
        auto_augment_policy=None,
        ra_magnitude=9,
        augmix_severity=3,
        random_erase_prob=0.0,
        backend="pil",
        use_v2=False,
        args = None,
        apply_mobius = True,
        apply_BGI = False,
        forward_mobius = False,
        mobius_prob = 0.2
    ):
        T = get_module(use_v2)

        transforms = []
        backend = backend.lower()
        if backend == "tensor":
            transforms.append(T.PILToTensor())
        elif backend != "pil":
            raise ValueError(f"backend can be 'tensor' or 'pil', but got {backend}")

        print(f'apply_mobius ', apply_mobius, 'apply_BGI ', apply_BGI, 'forward_mobius ', forward_mobius)
        print('auto_augment_policy ', auto_augment_policy)
        print(args)
        if apply_mobius:
            if False == apply_BGI:
                if forward_mobius:
                    transforms.append(MobiusTransform_Improved(p=mobius_prob)),
                    transforms.append(T.RandomResizedCrop(crop_size, interpolation=interpolation, antialias=True))
                else: 
                    transforms.append(T.RandomResizedCrop(crop_size, interpolation=interpolation, antialias=True))
                    transforms.append(MobiusTransform_Improved(p=mobius_prob))
            else:
                if forward_mobius:
                    transforms.append(MobiusTransform_Improved(p=mobius_prob, img_bck_ref = True))
                    transforms.append(T.RandomResizedCrop(crop_size, interpolation=interpolation, antialias=True))
                else: #experiment 1 (black background) - when Mobibus after randomResizeCrop
                    transforms.append(T.RandomResizedCrop(crop_size, interpolation=interpolation, antialias=True))
                    transforms.append(MobiusTransform_Improved(p=mobius_prob, img_bck_ref = True))
        else:
            transforms.append(T.RandomResizedCrop(crop_size, interpolation=interpolation, antialias=True))    
        
        if hflip_prob > 0:
            transforms.append(T.RandomHorizontalFlip(hflip_prob))
        if auto_augment_policy is not None:
            if auto_augment_policy == "ra":
                transforms.append(T.RandAugment(interpolation=interpolation, magnitude=ra_magnitude))
            elif auto_augment_policy == "ta_wide":
                transforms.append(T.TrivialAugmentWide(interpolation=interpolation))
            elif auto_augment_policy == "augmix":
                transforms.append(T.AugMix(interpolation=interpolation, severity=augmix_severity))
            else:
                aa_policy = T.AutoAugmentPolicy(auto_augment_policy)
                transforms.append(T.AutoAugment(policy=aa_policy, interpolation=interpolation))

        if backend == "pil":
            transforms.append(T.PILToTensor())

        transforms.extend(
            [
                T.ToDtype(torch.float, scale=True) if use_v2 else T.ConvertImageDtype(torch.float),
                T.Normalize(mean=mean, std=std),
            ]
        )
        if random_erase_prob > 0:
            transforms.append(T.RandomErasing(p=random_erase_prob))

        if use_v2:
            transforms.append(T.ToPureTensor())

        self.transforms = T.Compose(transforms)

    def __call__(self, img):
        return self.transforms(img)


class ClassificationPresetEval:
    def __init__(
        self,
        *,
        crop_size,
        resize_size=256,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        backend="pil",
        use_v2=False,
    ):
        T = get_module(use_v2)
        transforms = []
        backend = backend.lower()
        if backend == "tensor":
            transforms.append(T.PILToTensor())
        elif backend != "pil":
            raise ValueError(f"backend can be 'tensor' or 'pil', but got {backend}")

        transforms += [
            T.Resize(resize_size, interpolation=interpolation, antialias=True),
            T.CenterCrop(crop_size),
            
        ]
        

        if backend == "pil":
            transforms.append(T.PILToTensor())

        transforms += [
            T.ToDtype(torch.float, scale=True) if use_v2 else T.ConvertImageDtype(torch.float),
            T.Normalize(mean=mean, std=std),
        ]

        if use_v2:
            transforms.append(T.ToPureTensor())

        self.transforms = T.Compose(transforms)

    def __call__(self, img):
        return self.transforms(img)
