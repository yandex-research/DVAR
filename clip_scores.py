import os
import random
import time
from collections import defaultdict

import PIL
import torch
from PIL import Image
from torch.nn.functional import cosine_similarity
from torchvision import transforms
from tqdm import tqdm
from transformers import BertTokenizer, CLIPTokenizer

import clip
import wandb
from templates import imagenet_templates_base


class CLIPEvaluator:
    def __init__(self, device, clip_model='ViT-B/32') -> None:
        self.device = device
        self.model, clip_preprocess = clip.load(clip_model, device=self.device)

        self.clip_preprocess = clip_preprocess
        # Un-normalize from [-1.0, 1.0] (generator output) to [0, 1].
        self.preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] +
                                             clip_preprocess.transforms[:2] +  # to match CLIP input scale assumptions
                                             clip_preprocess.transforms[4:])  # + skip convert PIL to tensor

    def tokenize(self, strings: list):
        return clip.tokenize(strings).to(self.device)

    @torch.no_grad()
    def encode_text(self, tokens: list) -> torch.Tensor:
        return self.model.encode_text(tokens)

    @torch.no_grad()
    def get_image_features(self, images) -> torch.Tensor:
        if isinstance(images[0], PIL.Image.Image):
            # images is a list of PIL Images
            images = torch.stack([self.clip_preprocess(image) for image in images]).to(self.device)
        else:
            # images is a tensor of [-1, 1] images
            images = self.preprocess(images).to(self.device)
        return self.model.encode_image(images)

    def get_text_features(self, text: str) -> torch.Tensor:

        tokens = clip.tokenize(text).to(self.device)

        text_features = self.encode_text(tokens)
        return text_features

    def img_to_img_similarity(self, src_images, generated_images, reduction=True):
        src_img_features = self.get_image_features(src_images)
        gen_img_features = self.get_image_features(generated_images)
        if reduction:
            return cosine_similarity(src_img_features, gen_img_features).mean()
        else:
            return cosine_similarity(src_img_features, gen_img_features)

    def txt_to_img_similarity(self, text, generated_images, reduction=True):
        text_features = self.get_text_features(text)
        gen_img_features = self.get_image_features(generated_images)

        if reduction:
            return cosine_similarity(text_features, gen_img_features).mean()
        else:
            return cosine_similarity(text_features, gen_img_features)


@torch.inference_mode()
def select_init(data_dir, tokenizer, strategy, logger="wandb"):
    start_time = time.time()
    if isinstance(tokenizer, BertTokenizer):
        tok_vocab = tokenizer.vocab
        tokenizer_name = "bert-tokenizer"
    else:
        assert isinstance(tokenizer, CLIPTokenizer)
        tok_vocab = tokenizer.get_vocab()
        tokenizer_name = "clip-tokenizer"

    if strategy == "random":
        token_id = random.choice(list(tok_vocab.values()))
        token = tokenizer.convert_ids_to_tokens(token_id)
    elif strategy in ["best", "worst"]:
        id_to_return = 0 if strategy == "best" else -1
        out_file = f"{data_dir}/{tokenizer_name}_scores"
        if os.path.exists(out_file):
            print("found precomputed scores")
            scores = torch.load(out_file)
        else:
            caption_templates = random.sample(imagenet_templates_base, 4)
            clip_eval = CLIPEvaluator(device="cuda")
            is_image = lambda x: x.endswith(".jpg") or x.endswith(".png") or x.endswith("jpeg")
            images = [Image.open(f"{data_dir}/{x}") for x in os.listdir(data_dir) if is_image(x)]
            image_features = clip_eval.get_image_features(images).repeat(len(caption_templates), 1)  # [4 * n, dim]
            scores = {}

            for token in tqdm(tok_vocab):
                if tokenizer_name == "clip-tokenizer":
                    mod_token = token[:-len("</w>")] if token.endswith("</w>") else "##" + token
                    captions = [x.format(mod_token) for x in caption_templates]  # [4]
                else:
                    captions = [x.format(token) for x in caption_templates]  # [4]

                txt_features = clip_eval.get_text_features(captions).repeat(len(images), 1)  # [4 * n, dim]
                clip_txt_score = cosine_similarity(image_features, txt_features).mean()
                scores[token] = clip_txt_score.item()

            scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            torch.save(scores, out_file)

        token = scores[id_to_return][0]
        token_id = tokenizer.convert_tokens_to_ids(token)
    else:
        raise NotImplementedError(f"strategy {strategy} is not a valid choice."
                                  f" Possible choices are: best, worst, random, manual")
    print(f"Selected token: {token}")
    if logger == "wandb":
        end_time = time.time()
        wandb.config.update({"init_token": token, "init_selection_time": end_time - start_time})
    return token_id

