from pathlib import Path
import torchmetrics
from collections import defaultdict
from accelerate import Accelerator
from transformers import ViTImageProcessor, ViTModel
import torch
import tqdm
from torch.nn.functional import cosine_similarity
from PIL import Image

class DINOEvaluator:
    def __init__(self, device, dino_model='facebook/dino-vits16') -> None:
        self.device = device
        self.processor = ViTImageProcessor.from_pretrained('facebook/dino-vits16')
        self.model = ViTModel.from_pretrained('facebook/dino-vits16').to(device)

    @torch.inference_mode()
    def get_image_features(self, images) -> torch.Tensor:
        inputs = processor(images=images, return_tensors="pt").to(device=self.device)
        features = model(**inputs).last_hidden_state[:, 0, :]

    @torch.inference_mode()
    def img_to_img_similarity(self, src_images, generated_images, reduction=True):
        src_features = self.get_image_features(src_images)
        gen_features = self.get_image_features(src_images)

        return torchmetrics.functional.pairwise_cosine_similarity(src_features, gen_features).mean().item()

class DVIEvaluator:
    def __init__(self.device) -> None:
        self.model = lpips.LPIPS(net='alex')

    @torch.inference_mode()
    def get_score(images):
        score = 0
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                score += loss_fn_alex(images[0], images[1]).item()
        score /= len(images) * (len(images) - 1) / 2

        return score

