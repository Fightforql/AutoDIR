from typing import List, Tuple

import random
from PIL import Image, ImageFilter
import torchvision.transforms as T


class RandomGaussianBlur:
    def __init__(self, p: float = 0.0, radius_min: float = 0.1, radius_max: float = 2.0) -> None:
        self.p = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            radius = random.uniform(self.radius_min, self.radius_max)
            return img.filter(ImageFilter.GaussianBlur(radius))
        return img


def build_multi_view_augmentations(
    image_size: int,
    aug_views: int,
    color_jitter: float,
    gray_prob: float,
    blur_prob: float,
    crop_scale: Tuple[float, float],
) -> List[T.Compose]:
    normalize = T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))

    base = [
        T.RandomResizedCrop(image_size, scale=crop_scale, interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=color_jitter, contrast=color_jitter, saturation=color_jitter, hue=0.0),
        T.RandomGrayscale(p=gray_prob),
        RandomGaussianBlur(p=blur_prob),
        T.ToTensor(),
        normalize,
    ]

    views = [T.Compose(base) for _ in range(aug_views)]
    return views