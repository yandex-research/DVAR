import argparse
import math
import os
from typing import List

import numpy as np
import PIL
import torch
import torchvision
import wandb
from accelerate.logging import get_logger
from PIL import Image

logger = get_logger(__name__)


def dataset_log(path: str):
    images_paths = [os.path.join(path, file_path) for file_path in os.listdir(path)
                    if file_path.endswith(".jpg") or file_path.endswith(".png") or file_path.endswith(".jpeg")]
    images = [wandb.Image(_image_path) for _image_path in images_paths]
    wandb.log({"CONCEPT": images})


def save_progress(embedding, placeholder_token, logging_dir, name="learned_embeds.bin"):
    logger.info("Saving embeddings")
    base_path = os.path.join(logging_dir, "embeds")
    os.makedirs(base_path, exist_ok=True)
    learned_embeds_dict = {placeholder_token: embedding}
    torch.save(learned_embeds_dict, os.path.join(base_path, name))


def freeze_params(params):
    for param in params:
        param.requires_grad = False


def transform_img(img):
    tensor_img = torchvision.transforms.functional.pil_to_tensor(img)
    cropped_img = tensor_img / 127.5 - 1
    return cropped_img


def log_images(images: List[PIL.Image.Image], name: str = "", logging_dir: str = "", step=-1, cols=4):
    output_dir = os.path.join(logging_dir, "images", name)
    # log_locally:
    os.makedirs(output_dir, exist_ok=True)
    rows = math.ceil(len(images) / cols)
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, image in enumerate(images):
        if name != "train":
            image.save(os.path.join(output_dir, f"gs-{step}_{i}.jpg"))
        grid.paste(image, box=(i % cols * w, i // cols * h))
    grid.save(os.path.join(output_dir, f"gs-{step}_grid.jpg"))

    return grid


def sample(prompts, pipe, clip, logger, input, logging_dir,
           bs, sample_steps, guidance, log_unscaled, step=0, fp16=False):
    n_iters = math.ceil(len(prompts) / bs)
    samples_scaled = []
    reference_images = []

    with torch.inference_mode(), torch.autocast("cuda", enabled=fp16):
        for i in range(n_iters):
            generated = pipe(prompts[i * bs: (i + 1) * bs],
                             num_inference_steps=sample_steps, guidance_scale=guidance).images
            samples_scaled.extend(generated)
            reference_images.extend(input)

        reference_images = torch.stack(reference_images[:len(samples_scaled)])
        clip_img_score = clip.img_to_img_similarity(reference_images, samples_scaled)
        clip_txt_score = clip.txt_to_img_similarity(prompts, samples_scaled)

    grid = log_images(samples_scaled, step=step, name="scaled", logging_dir=logging_dir)
    if logger == "wandb":
        images_scaled = [wandb.Image(sample_, caption=prompts[idx]) for idx, sample_ in enumerate(samples_scaled)]
        wandb.log({
            "samples_scaled": images_scaled,
            "clip_img_score": clip_img_score,
            "clip_txt_score": clip_txt_score,
        }, step=step)
    else:
        logger.add_image("scaled", np.array(grid).transpose(2, 0, 1), step)
        logger.add_scalar("clip_img_score", clip_img_score, step)
        logger.add_scalar("clip_txt_score", clip_txt_score, step)

    if log_unscaled:
        samples = pipe(prompts[:bs], num_inference_steps=sample_steps, guidance_scale=0).images
        grid = log_images(samples, step=step, name="unscaled", logging_dir=logging_dir)

        if logger == "wandb":
            images = [wandb.Image(sample_, caption=prompts[idx]) for idx, sample_ in enumerate(samples)]
            wandb.log({"samples_unscaled": images}, step=step)
        else:
            logger.add_image("unscaled", np.array(grid).transpose(2, 0, 1), step)

    return clip_img_score


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--model", "-m",
        required=True,
        choices=["sd", "ldm"],
        help="Model for the pipeline: Stable Diffusion or Latent Diffusion"
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        required=True,
        help="A folder containing the training data."
    )
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default=None,
        required=True,
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument(
        "--initializer_token",
        type=str,
        help="A token to use as initializer word.",
        default='object',
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=50,
        help="Save learned_embeds.bin every X updates steps.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument("--learnable_property", type=str, default="object", help="Choose between 'object' and 'style'")
    parser.add_argument("--repeats", type=int, default=100, help="How many times to repeat the training data.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=4, help="Size of the evaluation batch."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=5000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=True,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adamw", "sam"],
                        help="Type of optimizer used for training")
    parser.add_argument("--save_pipeline", default=False, action="store_true",
                        help="Add this if you want to save the whole pipeline and not only learned embeddings")
    parser.add_argument("--pipeline_output_dir", default="", type=str,
                        help="By default equals to output_dir, where all learned embeddings will be saved")
    parser.add_argument("--save_init_embeds", default=True, action="store_true",
                        help="Sanity check for embedding initialization")
    parser.add_argument("--log_unscaled", default=False, action="store_true",
                        help="Whether to generate and save samples from unconditional generation")
    parser.add_argument("--sample_before_start", action="store_true",
                        help="Disable if you don't need a sanity check for sampling with "
                             "initial embeddings initialization")
    parser.add_argument("--sample_frequency", default=50, type=int, help="frequency of samples generation and logging")
    parser.add_argument("--sample_steps", default=200, type=int, help="number of DDIM sampler steps during generation")
    parser.add_argument("--sampling_seed", default=37, type=int,
                        help="fixed seed for sampling, different from training in order "
                             "not to train on the same examples")
    parser.add_argument("--template_set", default="default", type=str,
                        help="Strategy to select captions templates. Possible choices are: "
                             "default (imagenet_templates_base), one (only a{}), top-k (best templates for each image)")
    parser.add_argument("--n_val_prompts", default=8, type=int)
    parser.add_argument("--init_strategy", default="manual", choices=["manual", "best", "worst", "random"],
                        help="strategy to select initial word embedding. "
                             "If not manual --initializer_token argument is ignored")
    parser.add_argument("--guidance", default=7.5, type=float, help="Coefficient for classifier free guidance")
    parser.add_argument("--sam_adaptive", default=False, type=bool, help="Whether to use adaptive SAM")
    parser.add_argument("--sam_rho", default=0.05, type=float, help="Rho for SAM optimizer")
    parser.add_argument("--sam_momentum", default=0.9, type=float, help="Momentum for base optimizer in SAM")
    parser.add_argument("--offline_mode", action="store_true",
                        help="Flag for running without access to the Internet")
    parser.add_argument("--exp_code", default="11111",
                        help="Each 1/0 stands for one parameter among images/latents/noise/captions/timesteps being "
                             "fixed/unfixed in the eval batch")
    parser.add_argument("--logger", default="tensorboard", type=str, help="Which logger to use")
    parser.add_argument("--wandb",
                        help="Entity to use to write results into wandb")
    parser.add_argument("--project_name", default="", help="wandb project name")
    parser.add_argument("--exp_name", help="Experiment name for wandb", default="")
    parser.add_argument(
        "--eval_gradient_accumulation_steps",
        type=int,
        default=1,
        help="Used when an eval batch of the desired size does not fit into available VRAM.",
    )
    parser.add_argument("--variant", choices=["vanilla", "clip_early_stopping", "dvar_early_stopping", "short_iters"],
                        help="vanilla trains for fixed n_iters without intermediate clip score calculations, "
                             "clip_early_stopping evaluates intermediate results and stops training if no improve, "
                             "short_iters runs for predetermined number of iterations without intermediate sampling, "
                             "then stops and evaluates final embedding."
                             "Ours method doesn't sample at all and stops by variance early stopping criteria",
                        default="vanilla"
                        )
    parser.add_argument("--early_stop_eps", type=float, default=0.15,
                        help="change lower this value is not considered as a significant improvement")
    parser.add_argument("--early_stop_patience", type=int, default=200,
                        help="amount of consequent measurements during which no significant improvement was observed")
    parser.add_argument("--early_stop_freq", type=int, default=1,
                        help="early stopping loss calculation frequency")
    parser.add_argument("--mean_noise", action="store_true",
                        help="if used, fixed noise equals its expectation, i.e. null vector")
    parser.add_argument("--mean_latent", action="store_true", help="if used, fixed latent equals its expectation")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.train_data_dir is None:
        raise ValueError("You must specify a train data directory.")

    return args
