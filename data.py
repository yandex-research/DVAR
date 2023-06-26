import os
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from PIL.Image import Resampling
from torch.utils.data import Dataset
from torchvision import transforms

from templates import imagenet_style_templates_small, imagenet_templates_base



def preprocess(image, scale, resample):
    image = image.resize((scale, scale), resample=resample)
    image = np.array(image).astype(np.uint8)
    image = (image / 127.5 - 1.0).astype(np.float32)
    return image


def collate_fn(examples, with_prior_preservation):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    mask = [example["mask"] for example in examples]
    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]
        mask += [example["class_mask"] for example in examples]
    torch.save(examples, 'example_cd')
    input_ids = torch.cat(input_ids, dim=0)
    pixel_values = torch.stack(pixel_values)
    mask = torch.stack(mask)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    mask = mask.to(memory_format=torch.contiguous_format).float()

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "mask": mask.unsqueeze(1)
    }
    return batch


def is_image(str_path):
    return str_path.endswith(".jpg") or str_path.endswith(".png") or str_path.endswith("jpeg")


class TextualInversionDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        learnable_property="object",  # [object, style]
        size=512,
        repeats=100,
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        center_crop=False,
        with_prior_preservation=False,
        random_rescaling=False,
        concepts_list=None,
        num_class_images=200,
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p

        self.with_prior_preservation = with_prior_preservation
        self.random_rescaling = random_rescaling
        if with_prior_preservation:
            self.instance_prompts = []
            self.instance_images_path = []
            self.class_images_path = []
            for concept in concepts_list:
                self.instance_prompts.append(concept["instance_prompt"])
                inst_img_path = sorted([x for x in Path(concept["instance_data_dir"]).iterdir() if is_image(str(x))])
                self.instance_images_path.extend(inst_img_path)

                class_data_root = Path(concept['class_data_dir'])
                if os.path.isdir(class_data_root):
                    class_images_path = list(class_data_root.iterdir())
                    class_prompt = [concept["class_prompt"] for _ in range(len(class_images_path))]
                else:
                    with open(class_data_root, "r") as f:
                        class_images_path = f.read().splitlines()
                    with open(concept["class_prompt"], "r") as f:
                        class_prompt = f.read().splitlines()
                class_img_path = [(x, y) for (x, y) in zip(class_images_path, class_prompt)]
                self.class_images_path.extend(class_img_path[:num_class_images])
            self.num_class_images = len(self.class_images_path)
        else:
            self.instance_images_path = sorted([os.path.join(self.data_root, file_path)
                                                for file_path in os.listdir(self.data_root) if is_image(file_path)])
        self.num_instance_images = len(self.instance_images_path)
        self._length = self.num_instance_images

        if set == "train":
            self._length = self.num_instance_images * repeats

        self.interpolation = Resampling.BILINEAR if with_prior_preservation else Resampling.BICUBIC

        if not random_rescaling:
            # preprocess all images
            self.images = []

            for image_path in self.instance_images_path:
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
        elif learnable_property == "object":
            self.templates = imagenet_templates_base
        else:
            raise ValueError(f"{learnable_property} is not supported")

        formatted_templates = [template.format(self.placeholder_token) for template in self.templates]
        self.tokenized_templates = self.tokenizer(formatted_templates,
                                                  padding="max_length",
                                                  truncation=True,
                                                  max_length=self.tokenizer.model_max_length,
                                                  return_tensors="pt",
                                                  ).input_ids

        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)
        self.image_transforms = transforms.Compose(
            [
                self.flip_transform,
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        if self.with_prior_preservation:
            instance_image = self.instance_images_path[i % self.num_instance_images]
            instance_prompt = self.instance_prompts[0]
            instance_image = Image.open(instance_image)
            if not instance_image.mode == "RGB":
                instance_image = instance_image.convert("RGB")
            instance_image = self.flip_transform(instance_image)

            if self.random_rescaling:
                ##############################################################################
                #### apply resize augmentation and create a valid image region mask ##########
                ##############################################################################
                if torch.randint(0, 3, (1,)) < 2:
                    random_scale = torch.randint(self.size // 3, self.size + 1, (1,)).item()
                else:
                    random_scale = torch.randint(int(1.2 * self.size), int(1.4 * self.size), (1,)).item()

                if random_scale % 2 == 1:
                    random_scale += 1
                if random_scale < 0.6 * self.size:
                    add_to_caption = ["a far away ", "very small "][torch.randint(0, 2, (1,))]
                    instance_prompt = add_to_caption + instance_prompt
                    cx = torch.randint(random_scale // 2, self.size - random_scale // 2 + 1, (1,))
                    cy = torch.randint(random_scale // 2, self.size - random_scale // 2 + 1, (1,))
                    instance_image1 = preprocess(instance_image, random_scale, self.interpolation)
                    instance_image = np.zeros((self.size, self.size, 3), dtype=np.float32)
                    instance_image[cx - random_scale // 2: cx + random_scale // 2,
                    cy - random_scale // 2: cy + random_scale // 2, :] = instance_image1

                    mask = np.zeros((self.size // 8, self.size // 8))
                    mask[(cx - random_scale // 2) // 8 + 1: (cx + random_scale // 2) // 8 - 1,
                    (cy - random_scale // 2) // 8 + 1: (cy + random_scale // 2) // 8 - 1] = 1.
                elif random_scale > self.size:
                    add_to_caption = ["zoomed in ", "close up "][torch.randint(0, 2, (1,))]
                    instance_prompt = add_to_caption + instance_prompt
                    cx = torch.randint(self.size // 2, random_scale - self.size // 2 + 1, (1,))
                    cy = torch.randint(self.size // 2, random_scale - self.size // 2 + 1, (1,))

                    instance_image = preprocess(instance_image, random_scale, self.interpolation)
                    instance_image = instance_image[cx - self.size // 2: cx + self.size // 2,
                                     cy - self.size // 2: cy + self.size // 2, :]
                    mask = np.ones((self.size // 8, self.size // 8))
                else:
                    instance_image = preprocess(instance_image, self.size, self.interpolation)
                    mask = np.ones((self.size // 8, self.size // 8))
            else:
                instance_image = preprocess(instance_image, self.size, self.interpolation)
                mask = np.ones((self.size // 8, self.size // 8))
            example["mask"] = torch.from_numpy(mask)
            example["class_mask"] = torch.ones_like(example["mask"])

            example["instance_images"] = torch.from_numpy(instance_image).permute(2, 0, 1)
            example["instance_prompt_ids"] = self.tokenizer(
                instance_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

            class_image, class_prompt = self.class_images_path[i % self.num_class_images]
            class_image = Image.open(class_image)
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                class_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids
        else:
            image = self.images[i % self.num_instance_images]
            tokenized_template = self.tokenized_templates[i % len(self.tokenized_templates)]
            
            example["input_ids"] = tokenized_template
            example["pixel_values"] = self.flip_transform(image)
        return example
