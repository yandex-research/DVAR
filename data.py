from collections import OrderedDict
import os

import torch
import torchvision
from PIL import Image
from PIL.Image import Resampling
from torch.utils.data import Dataset
from torchvision import transforms

from clip_scores import select_best_templates
from templates import imagenet_templates_base, imagenet_style_templates_small


class TextualInversionDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        learnable_property="object",  # [object, style]
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        center_crop=False,
        template_set="default",
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p

        is_image = lambda x: x.endswith(".jpg") or x.endswith(".png") or x.endswith("jpeg")
        self.image_paths = sorted([os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)
                                   if is_image(file_path)])

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "bilinear": Resampling.BILINEAR,
            "bicubic": Resampling.BICUBIC,
            "lanczos": Resampling.LANCZOS,
        }[interpolation]

        # preprocess all images
        self.images = []

        for image_path in self.image_paths:
            image = Image.open(image_path)
            if not image.mode == "RGB":
                image = image.convert("RGB")

            # default to score-sde preprocessing
            if center_crop:
                image = torchvision.transforms.functional.center_crop(image, min(image.size))
            resized_image = image.resize((self.size, self.size), resample=self.interpolation)
            torch_image = torchvision.transforms.functional.to_tensor(resized_image)
            torch_image = torch_image * 2 - 1

            self.images.append(torch_image)

        if learnable_property == "style":
            self.templates = imagenet_style_templates_small
        elif template_set == "default":
            self.templates = imagenet_templates_base
        elif template_set == "one":
            self.templates = ['a {}']
        elif template_set.startswith("top-"):
            k = int(template_set.split("-")[-1])
            self.templates = select_best_templates(self.image_paths, k)
        else:
            raise ValueError(f"{template_set} is not supported")

        placeholder_string = self.placeholder_token
        if isinstance(self.templates, dict):
            # if the best templates were selected for each image
            formatted_templates = {
                img_path: (template_for_img.format(placeholder_string) for template_for_img in templates_for_img)
                for img_path, templates_for_img in self.templates.items()
            }
            self.tokenized_templates = {
                img_path: [self.tokenizer(template,
                                          padding="max_length",
                                          truncation=True,
                                          max_length=self.tokenizer.model_max_length,
                                          return_tensors="pt",
                                          ).input_ids[0]
                           for template in templates_for_img]
                for img_path, templates_for_img in formatted_templates.items()
            }
        else:
            formatted_templates = (template.format(placeholder_string) for template in self.templates)
            self.tokenized_templates = [
                self.tokenizer(template,
                               padding="max_length",
                               truncation=True,
                               max_length=self.tokenizer.model_max_length,
                               return_tensors="pt",
                               ).input_ids[0]
                for template in formatted_templates
            ]

        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        img_path = self.image_paths[i % self.num_images]
        image = self.images[i % self.num_images]

        if isinstance(self.templates, dict):
            rand_ind = torch.randint(len(self.templates[img_path]), (1,))
            tokenized_template = self.tokenized_templates[img_path][rand_ind]
        else:
            rand_ind = torch.randint(len(self.templates), (1,))
            tokenized_template = self.tokenized_templates[rand_ind]

        example["input_ids"] = tokenized_template
        example["pixel_values"] = self.flip_transform(image)
        return example
