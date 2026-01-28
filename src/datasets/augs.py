import random
import logging

from PIL import Image
import torch
from torchvision import transforms
import src.utils as utils

log = logging.getLogger(__name__)


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number,
                 initial_crop_scale=None, global_crop_resolution=(224, 224), local_crop_resolution=(96, 96),
                 motion_crop_scale=None, motion_crop_resolution=None, motion_crop_ratio=None,
                 motion_crop_jitter=None, motion_crop_aug=True, motion_affine_aug_params={}, visualize=False,
                 ):
        color_jitter = transforms.Compose([
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        self.normalize = normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.initial_crop_scale = initial_crop_scale
        self.global_crop_resolution = global_crop_resolution
        self.global_crops_scale = global_crops_scale

        # first global transform
        self.global_transfo1 = transforms.Compose([
            # transforms.RandomResizedCrop((512, 1024), ratio=(1.5, 2.5), scale=global_crops_scale, interpolation=Image.BICUBIC),
            color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global transform
        self.global_transfo2 = transforms.Compose([
            # transforms.RandomResizedCrop((512, 1024), ratio=(1.5, 2.5), scale=global_crops_scale, interpolation=Image.BICUBIC),
            color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(local_crop_resolution, scale=local_crops_scale, interpolation=Image.BICUBIC),
            # transforms.RandomResizedCrop((224, 448), ratio=(1.5, 2.5), scale=local_crops_scale, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

        self.motion_crop_scale = motion_crop_scale
        self.motion_crop_resolution = motion_crop_resolution
        self.motion_crop_ratio = motion_crop_ratio
        self.motion_crop_jitter = motion_crop_jitter
        self.motion_crop_aug = motion_crop_aug

        self.motion_affine_aug = None
        if motion_affine_aug_params:
            self.motion_affine_aug = transforms.RandomAffine(**motion_affine_aug_params)

        self.visualize = visualize

    def __call__(self, image_x, image_y=None):
        crops = []
        paired_frame = image_y is not None
        
        if paired_frame and self.motion_crop_scale is not None:
            i, j, h, w  = transforms.RandomResizedCrop.get_params(image_x, self.motion_crop_scale, self.motion_crop_ratio)
            motion_crop_x_params = (i, j, h, w)
            if self.motion_crop_jitter is not None:
                # motion crop jitter is a percentage of the h, w of the crop
                H, W = image_x.size
                h_jitter_min = int(h * self.motion_crop_jitter[0])
                h_jitter_max = int(h * self.motion_crop_jitter[1])
                w_jitter_min = int(w * self.motion_crop_jitter[0])
                w_jitter_max = int(w * self.motion_crop_jitter[1])
                h_jitter = random.randint(h_jitter_min, h_jitter_max)
                w_jitter = random.randint(w_jitter_min, w_jitter_max)
                if random.random() < 0.5:
                    h_jitter = -h_jitter
                if random.random() < 0.5:
                    w_jitter = -w_jitter
                i = max(min(i + h_jitter, H - h), 0)
                j = max(min(j + w_jitter, W - w), 0)
            motion_crop_y_params = (i, j, h, w)
            motion_crop_x = transforms.functional.resized_crop(image_x, *motion_crop_x_params, self.motion_crop_resolution, interpolation=Image.BICUBIC)
            motion_crop_y = transforms.functional.resized_crop(image_y, *motion_crop_y_params, self.motion_crop_resolution, interpolation=Image.BICUBIC)
            mask_x = None
            mask_y = None
            if self.motion_affine_aug is not None:
                motion_crop_x = transforms.functional.to_tensor(motion_crop_x)
                motion_crop_y = transforms.functional.to_tensor(motion_crop_y)
                H, W = motion_crop_x.shape[-2:]
                mask_x = torch.ones((1, H, W))
                mask_y = torch.ones((1, H, W))
                motion_crop_x = self.motion_affine_aug(torch.cat([motion_crop_x, mask_x], dim=0))
                motion_crop_y = self.motion_affine_aug(torch.cat([motion_crop_y, mask_y], dim=0))
                motion_crop_x, mask_x = motion_crop_x[:-1], motion_crop_x[-1:]
                motion_crop_y, mask_y = motion_crop_y[:-1], motion_crop_y[-1:]
                motion_crop_x = transforms.functional.to_pil_image(motion_crop_x)
                motion_crop_y = transforms.functional.to_pil_image(motion_crop_y)
            if self.visualize:
                crops.append(transforms.functional.to_tensor(motion_crop_x))
                crops.append(transforms.functional.to_tensor(motion_crop_y))
            elif mask_x is not None:
                crops.append(mask_x)
                crops.append(mask_y)
            if self.motion_crop_aug:
                motion_crop_x = self.global_transfo1(motion_crop_x)
                motion_crop_y = self.global_transfo2(motion_crop_y)
            else:
                motion_crop_x = self.normalize(motion_crop_x)
                motion_crop_y = self.normalize(motion_crop_y)
            crops.append(motion_crop_x)
            crops.append(motion_crop_y)

        if self.initial_crop_scale is not None:
            initial_crop_params = transforms.RandomResizedCrop.get_params(image_x, self.initial_crop_scale, (3./4, 4./3))
            image_x = transforms.functional.crop(image_x, *initial_crop_params) # (3, H', W')
            if paired_frame:
                image_y = transforms.functional.crop(image_y, *initial_crop_params) # (3, H', W')
        
        global_crop_params1 = transforms.RandomResizedCrop.get_params(image_x, self.global_crops_scale, (3./4, 4./3))
        global_crop_params2 = transforms.RandomResizedCrop.get_params(image_x, self.global_crops_scale, (3./4, 4./3))

        global_crop_1_x = transforms.functional.resized_crop(image_x, *global_crop_params1, self.global_crop_resolution, interpolation=Image.BICUBIC)
        global_crop_2_x = transforms.functional.resized_crop(image_x, *global_crop_params2, self.global_crop_resolution, interpolation=Image.BICUBIC)

        if paired_frame:
            global_crop_1_y = transforms.functional.resized_crop(image_y, *global_crop_params1, self.global_crop_resolution, interpolation=Image.BICUBIC)
            global_crop_2_y = transforms.functional.resized_crop(image_y, *global_crop_params2, self.global_crop_resolution, interpolation=Image.BICUBIC)

        # With 50% probability, flip global crops 1 and 2
        if random.random() < 0.5:
            global_crop_1_x = transforms.functional.hflip(global_crop_1_x)
            if paired_frame:
                global_crop_1_y = transforms.functional.hflip(global_crop_1_y) 

        if random.random() < 0.5:
            global_crop_2_x = transforms.functional.hflip(global_crop_2_x)
            if paired_frame:
                global_crop_2_y = transforms.functional.hflip(global_crop_2_y)

        if self.visualize:
            crops.append(transforms.functional.to_tensor(global_crop_1_x))
            crops.append(transforms.functional.to_tensor(global_crop_2_x))
            if paired_frame:
                crops.append(transforms.functional.to_tensor(global_crop_1_y))
                crops.append(transforms.functional.to_tensor(global_crop_2_y))

        crops.append(self.global_transfo1(global_crop_1_x))
        crops.append(self.global_transfo2(global_crop_2_x))
        if paired_frame:
            crops.append(self.global_transfo1(global_crop_1_y))
            crops.append(self.global_transfo2(global_crop_2_y))

        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image_x))
        if paired_frame:
            for _ in range(self.local_crops_number):
                crops.append(self.local_transfo(image_y))

        return crops
