import datetime
import itertools
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import (AutoencoderKL, DDIMScheduler, LDMTextToImagePipeline,
                       PNDMScheduler, StableDiffusionPipeline,
                       UNet2DConditionModel)
from diffusers.optimization import get_scheduler
from PIL import Image
from PIL.Image import Resampling
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import BertTokenizer, CLIPTokenizer
from lora_diffusion import inject_trainable_lora, extract_lora_ups_down

import wandb
from clip_scores import CLIPEvaluator, select_init
from custom_diffusion import retrieve, create_custom_diffusion, CustomDiffusionPipeline
from data import TextualInversionDataset, collate_fn
from early_stopping import ClipEarlyStopper, VarEarlyStopper
from evaluate import concept_prompts
from optimizers import get_optimizer
from templates import imagenet_templates_base
from text_emb import (TextualInversionCLIPTextModel,
                      TextualInversionLDMBertModel)
from utils import evaluate, freeze_params, log_images, logger, parse_args, save_progress


def open_image(image_path, resolution=512):
    image = Image.open(image_path).resize((resolution, resolution), resample=Resampling.BICUBIC)
    if not image.mode == 'RGB':
        image = image.convert("RGB")
    return image


def main():
    args = parse_args()
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    logging_dir = f"{args.output_dir}/{args.logging_dir}-{now}"
    args.with_prior_preservation = args.method == 'custom'
    if not args.pretrained_model_name_or_path:
        if args.method == 'custom' and args.model == 'sd':
            args.pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4"
        elif args.model == 'sd':
            args.pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
        elif args.model == 'ldm':
            "CompVis/ldm-text2im-large-256"
        else:
            raise NotImplementedError(args.model)

    if not args.resolution:
        args.resolution = 256 if args.model == "ldm" else 512
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    args.fp16 = args.mixed_precision == 'fp16'
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    use_auth_token = not args.offline_mode
    # Handle the repository creation
    args.concept = Path(args.train_data_dir).name
    if accelerator.is_main_process:
        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initialize automatically on the main process.
        if args.logger == "tensorboard":
            stat_logger = SummaryWriter(logging_dir)
        elif args.logger == "wandb":
            stat_logger = "wandb"
            project_name = args.project_name if args.project_name else "dvar_inversion"
            wandb.init(entity=args.wandb, project=project_name)
            if args.exp_name:
                wandb.run.name = args.exp_name
            else:
                wandb.run.name = f"{args.variant}-{args.concept}-{args.model}-{args.init_strategy}-{args.optimizer}"

    accelerator.init_trackers("")

    # Load the tokenizer and add the placeholder token as an additional special token
    # Load models and create wrapper for stable diffusion
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path,
                                                subfolder="unet", use_auth_token=use_auth_token)
    if args.method == 'dreambooth':
        unet_lora_params, train_names = inject_trainable_lora(unet)
    
    if args.model == "sd":
        if args.tokenizer_name:
            tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
        elif args.pretrained_model_name_or_path:
            tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path,
                                                      subfolder="tokenizer", use_auth_token=use_auth_token)
        text_encoder = TextualInversionCLIPTextModel.from_pretrained(args.pretrained_model_name_or_path,
                                                                     subfolder="text_encoder",
                                                                     use_auth_token=use_auth_token)
        vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path,
                                            subfolder="vae", use_auth_token=use_auth_token)
        noise_scheduler = PNDMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler",
                                                    use_auth_token=use_auth_token)
        if args.method != 'custom':
            pipeline = StableDiffusionPipeline(
                text_encoder=accelerator.unwrap_model(text_encoder),
                vae=vae,
                unet=unet,
                tokenizer=tokenizer,
                scheduler=noise_scheduler,
                safety_checker=None,
                feature_extractor=None,
            )
    else:
        if args.tokenizer_name:
            tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name)
        elif args.pretrained_model_name_or_path:
            tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_name_or_path,
                                                      subfolder="tokenizer", use_auth_token=use_auth_token)
        text_encoder = TextualInversionLDMBertModel.from_pretrained(args.pretrained_model_name_or_path,
                                                                    subfolder="bert")
        max_position_ids = text_encoder.model.embed_positions.weight.size()[0]
        vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path,
                                            subfolder="vqvae", use_auth_token=use_auth_token)
        noise_scheduler = DDIMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler")
        if args.method != 'custom':
            pipeline = LDMTextToImagePipeline(
                vqvae=vae,
                bert=accelerator.unwrap_model(text_encoder),
                unet=unet,
                tokenizer=tokenizer,
                scheduler=noise_scheduler,
            )
    scaling_factor = vae.config.scaling_factor

    if args.init_strategy == "manual":
        # Convert the initializer_token, placeholder_token to ids
        token_ids = tokenizer.encode([args.initializer_token], add_special_tokens=False)
        # Check if initializer_token is a single token or a sequence of tokens
        if len(token_ids) > 1:
            raise ValueError("The initializer token must be a single token.")

        initializer_token_id = token_ids[0]
    else:
        initializer_token_id = select_init(args.train_data_dir, tokenizer,
                                           strategy=args.init_strategy, logger=stat_logger)

    # Add the placeholder token in tokenizer
    num_added_tokens = tokenizer.add_tokens(args.placeholder_token)
    if num_added_tokens == 0:
        raise ValueError(
            f"The tokenizer already contains the token {args.placeholder_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )
    placeholder_token_id = tokenizer.convert_tokens_to_ids(args.placeholder_token)

    # Initialise the newly added placeholder token with the embeddings of the initializer token
    token_embeds = text_encoder.get_input_embeddings().weight.data
    text_encoder.patch_emb(token_embeds[initializer_token_id])
    original_embed = token_embeds[initializer_token_id].to(accelerator.device)
    learned_embed = text_encoder.get_input_embeddings().concept_token.data

    if args.method == 'custom':
        pipeline = CustomDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=unet,
            text_encoder=text_encoder,
            modifier_token=[args.placeholder_token],
            modifier_token_id=[placeholder_token_id],
            safety_checker=None,
        ).to(accelerator.device)
    if args.save_pipeline:
        if not args.pipeline_output_dir:
            args.pipeline_output_dir = args.output_dir
        pipeline.save_pretrained(args.pipeline_output_dir)

    # Freeze vae and unet
    freeze_params(vae.parameters())
    if args.method == 'inversion':
        freeze_params(unet.parameters())
    elif args.method == 'custom':
        unet = create_custom_diffusion(unet, 'crossattn_kv')

    # Freeze all parameters except for the token embeddings in text encoder
    if args.model == "sd":
        params_to_freeze = itertools.chain(
            text_encoder.text_model.encoder.parameters(),
            text_encoder.text_model.final_layer_norm.parameters(),
            text_encoder.text_model.embeddings.position_embedding.parameters(),
        )
    else:
        params_to_freeze = itertools.chain(
            text_encoder.model.embed_positions.parameters(),
            text_encoder.model.layers.parameters(),
            text_encoder.to_logits.parameters(),
        )
    freeze_params(params_to_freeze)

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
    if args.with_prior_preservation:
        args.learning_rate = args.learning_rate * 2.

    # Initialize the optimizer
    args.optimizer = args.optimizer.lower()
    optimizer_params = {
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'adam_beta1': args.adam_beta1,
        'adam_beta2': args.adam_beta2,
        'adam_epsilon': args.adam_epsilon,
        'sam_momentum': args.sam_momentum,
        'sam_rho': args.sam_rho,
        'sam_adaptive': args.sam_adaptive,
    }
    if args.method == 'inversion':
        params_to_optimize = text_encoder.get_input_embeddings().parameters()
    elif args.method == 'custom':
        params_to_optimize = itertools.chain(text_encoder.get_input_embeddings().parameters(),
                                             [x[1] for x in unet.named_parameters() if
                                              ('attn2.to_k' in x[0] or 'attn2.to_v' in x[0])])
    elif args.method == 'dreambooth':
        params_to_optimize = itertools.chain(text_encoder.get_input_embeddings().parameters(),
                                            *unet_lora_params)

    optimizer = get_optimizer(params=params_to_optimize,
                              name=args.optimizer, opt_args=optimizer_params)
    if args.with_prior_preservation:
        # custom diffusion (and probably dreambooth) stuff
        assert args.class_data_dir is not None
        random.seed(args.seed)
        args.class_prompt = " ".join(args.class_prompt.split("_"))
        random_template = random.choice(imagenet_templates_base)
        instance_prompt = f"{args.placeholder_token} {args.class_prompt}"
        args.concepts_list = [
            {
                "instance_prompt": random_template.format(instance_prompt),
                "class_prompt": args.class_prompt,
                "instance_data_dir": args.train_data_dir,
                "class_data_dir": args.class_data_dir
            }
        ]

        for i, concept in enumerate(args.concepts_list):
            class_images_dir = Path(concept['class_data_dir'])
            if not class_images_dir.exists():
                class_images_dir.mkdir(parents=True, exist_ok=True)
            if accelerator.is_main_process:
                name = '_'.join(concept['class_prompt'].split())
                if not Path(os.path.join(class_images_dir, name)).exists() or len(
                        list(Path(os.path.join(class_images_dir, name)).iterdir())) < args.num_class_images:
                    print(concept['class_prompt'], class_images_dir, args.num_class_images)
                    retrieve(concept['class_prompt'], class_images_dir, args.num_class_images)
            concept['class_prompt'] = os.path.join(class_images_dir, 'caption.txt')
            concept['class_data_dir'] = os.path.join(class_images_dir, 'images.txt')
            args.concepts_list[i] = concept
            accelerator.wait_for_everyone()
    else:
        args.concepts_list = None

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    train_dataset = TextualInversionDataset(
        data_root=args.train_data_dir,
        tokenizer=tokenizer,
        size=args.resolution,
        placeholder_token=args.placeholder_token,
        repeats=args.repeats,
        learnable_property=args.learnable_property,
        center_crop=args.center_crop,
        set="train",
        concepts_list=args.concepts_list,
        with_prior_preservation=args.with_prior_preservation,
        random_rescaling=args.method == 'custom',
        num_class_images=args.num_class_images,
    )

    if args.with_prior_preservation:
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                                                       collate_fn=lambda examples: collate_fn(examples, True))
        eval_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.eval_batch_size, shuffle=False,
                                                      collate_fn=lambda examples: collate_fn(examples, False))
    else:
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
        eval_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.eval_batch_size, shuffle=False)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    text_encoder, vae, unet, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        text_encoder, vae, unet, train_dataloader, eval_dataloader, lr_scheduler
    )
    eval_dataloader_iter = iter(eval_dataloader)
    if args.optimizer != "sam":
        optimizer = accelerator.prepare(optimizer)

    # Keep vae and unet in eval model as we don't train these
    vae.eval()
    unet.eval()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # log train data
    if accelerator.is_main_process:
        os.makedirs(logging_dir)
        train_images = [open_image(x, args.resolution) for x in train_dataset.instance_images_path]
        log_images(train_images, prompts=[args.placeholder_token for _ in range(len(train_images))],
                   name='train', logging_dir=logging_dir, step=0, logger=stat_logger)

    args.early_stopping = args.variant in ["dvar_early_stopping", "clip_early_stopping"]

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0

    train_prompts = [x.format(args.placeholder_token) for x in
                     np.random.choice(imagenet_templates_base, args.n_train_prompts)]
    val_prompts = args.validation_prompts.split(',') if args.validation_prompts else concept_prompts.get(args.concept,
                                                                                                         [])
    val_prompts = [x.format(args.placeholder_token) for x in val_prompts]
    val_prompts = val_prompts * args.n_images_per_val_prompt
    reference_images = [train_images[i % len(train_images)] for i in range(max(args.n_train_prompts, len(val_prompts)))]

    # initialize clip for samples evaluation
    clip = CLIPEvaluator(device=accelerator.device, clip_model=args.clip_path)

    with open(f"{logging_dir}/train_prompts.txt", "w") as f:
        f.writelines("\n".join(train_prompts))

    # Create the pipeline using the trained modules
    object_to_save = {'inversion': learned_embed, 'custom': pipeline, 'dreambooth': (learned_embed, unet)}[args.method]
    if accelerator.is_main_process:
        pipeline.to(accelerator.device)
        if args.save_init_embeds:
            name = "initial_weights.bin"
            save_progress(object_to_save, args.placeholder_token, logging_dir, name=name, method=args.method)
        if args.sample_before_start:
            evaluate(pipeline, train_prompts, val_prompts, clip, reference_images, logging_dir, args.placeholder_token,
                     sample_steps=args.sample_steps, guidance=args.guidance, fp16=args.fp16, step=global_step,
                     sampling_seed=args.sampling_seed, log_unscaled=args.log_unscaled, logger=stat_logger)

    wandb.config.update(args)
    logs = {}

    if args.variant == "clip_early_stopping":
        early_stopper = ClipEarlyStopper(eps=args.early_stop_eps, patience=args.early_stop_patience)
    elif args.variant == "dvar_early_stopping":
        args.fixed_img, args.fixed_latents, args.fixed_noise, args.fixed_captions, args.fixed_timesteps = args.exp_code
        early_stopper = VarEarlyStopper(eps=args.early_stop_eps, window=args.early_stop_patience)

        with torch.inference_mode(), torch.random.fork_rng(devices=['cuda', 'cpu']):
            eval_batch = next(eval_dataloader_iter)
            if args.model == "ldm":
                eval_captions = eval_batch["input_ids"][:, :max_position_ids]
            else:
                eval_captions = eval_batch["input_ids"]
            eval_latents = vae.encode(eval_batch["pixel_values"]).latent_dist.sample().detach()
            eval_latents = eval_latents * scaling_factor
            eval_noise = torch.randn(eval_latents.shape).to(eval_latents.device)
            rand_perm = torch.randperm(1000).to(eval_latents.device)
            if args.triple:
                rand_perms_triple = {
                    "begin": torch.randperm(333, device=eval_latents.device, dtype=torch.long),
                    "middle": 333 + torch.randperm(333, device=eval_latents.device, dtype=torch.long),
                    "end": 666 + torch.randperm(333, device=eval_latents.device, dtype=torch.long),
                }
    else:
        early_stopper = None
    train_start = time.time()

    for epoch in range(args.num_train_epochs):
        text_encoder.train()
        if args.method == 'custom':
            unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(text_encoder):
                if (args.early_stopping and early_stopper.stopped) or global_step >= args.max_train_steps:
                    break
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"]).latent_dist.sample().detach()
                latents = latents * scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn(latents.shape, device=latents.device)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,),
                                          device=latents.device, dtype=torch.long)

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                if args.model == "ldm":
                    batch["input_ids"] = batch["input_ids"][:, :max_position_ids]
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                def closure():
                    optimizer.zero_grad()
                    if args.with_prior_preservation:
                        # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                        model_pred, model_pred_prior = torch.chunk(noise_pred, 2, dim=0)
                        target, target_prior = torch.chunk(noise, 2, dim=0)
                        mask = torch.chunk(batch["mask"], 2, dim=0)[0]
                        # Compute instance loss
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                        loss = ((loss * mask).sum([1, 2, 3]) / mask.sum([1, 2, 3])).mean()

                        # Compute prior loss
                        prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                        # Add the prior loss to the instance loss.
                        loss = loss + prior_loss
                    else:
                        loss = F.mse_loss(noise, noise_pred, reduction="none").mean()

                    logs['loss'] = loss.detach()
                    accelerator.backward(loss, retain_graph=True)

                    return loss

                closure()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    grad_norms = [torch.norm(x.grad, 2) for x in optimizer.param_groups[0]['params']
                                  if x.grad is not None]
                    logs['gradient_norm'] = torch.norm(torch.stack(grad_norms), 2).item()

                    if args.variant == "dvar_early_stopping" and global_step % args.early_stop_freq == 0:
                        eval_loss = 0
                        if args.triple:
                            triple_loss = {"begin": 0, "middle": 0, "end": 0}
                        with torch.inference_mode(), torch.random.fork_rng(devices=['cuda', 'cpu']):
                            if not int(args.fixed_timesteps):
                                rand_perm = torch.randperm(1000, device=eval_latents.device, dtype=torch.long)
                            for acc_step in range(args.eval_gradient_accumulation_steps):
                                if not int(args.fixed_img):
                                    # 00xxx option
                                    try:
                                        eval_images = next(eval_dataloader_iter)["pixel_values"]
                                    except StopIteration:
                                        eval_dataloader_iter = iter(eval_dataloader)
                                        eval_images = next(eval_dataloader_iter)["pixel_values"]
                                    # new images -> new_latents
                                    if args.mean_latent:
                                        eval_latents = vae.encode(
                                            eval_images).latent_dist.mean.detach() * scaling_factor
                                    else:
                                        eval_latents = vae.encode(
                                            eval_images).latent_dist.sample().detach() * scaling_factor
                                elif not int(args.fixed_latents):
                                    # 10xxx
                                    eval_latents = vae.encode(
                                        eval_images).latent_dist.sample().detach() * scaling_factor
                                elif args.mean_latent:
                                    # 11xxx and replace
                                    eval_latents = vae.encode(eval_images).latent_dist.mean.detach() * scaling_factor
                                # else 11xxx do nothing, we already have eval_latents for these eval_images

                                if not int(args.fixed_noise):
                                    eval_noise = torch.randn(eval_latents.shape, device=eval_latents.device)
                                elif args.mean_noise:
                                    eval_noise = torch.zeros(eval_latents.shape, device=eval_latents.device)

                                if not int(args.fixed_captions):
                                    try:
                                        eval_captions = next(eval_dataloader_iter)["input_ids"]
                                    except StopIteration:
                                        eval_dataloader_iter = iter(eval_dataloader)
                                        eval_captions = next(eval_dataloader_iter)["input_ids"]
                                    if args.model == "ldm":
                                        eval_captions = eval_captions[:, :max_position_ids]

                                eval_timesteps = rand_perm[args.eval_batch_size * acc_step:
                                                           args.eval_batch_size * (acc_step + 1)]

                                eval_noisy_latents = noise_scheduler.add_noise(eval_latents, eval_noise, eval_timesteps)

                                eval_hidden_states = text_encoder(eval_captions)[0]
                                eval_noise_pred = unet(eval_noisy_latents, eval_timesteps, eval_hidden_states).sample
                                eval_mse_loss = F.mse_loss(eval_noise, eval_noise_pred, reduction="mean")
                                eval_loss += eval_mse_loss.detach().cpu()
                                if args.triple:
                                    for interval in rand_perms_triple:
                                        triple_timesteps = rand_perms_triple[interval][args.eval_batch_size * acc_step:
                                                                   args.eval_batch_size * (acc_step + 1)]

                                        triple_noisy_latents = noise_scheduler.add_noise(eval_latents, eval_noise, triple_timesteps)

                                        triple_noise_pred = unet(triple_noisy_latents, triple_timesteps, eval_hidden_states).sample
                                        triple_mse_loss = F.mse_loss(eval_noise, triple_noise_pred, reduction="mean")
                                        triple_loss[interval] += triple_mse_loss.detach().cpu()
                            logs["eval_loss"] = eval_loss / args.eval_gradient_accumulation_steps
                            logs["normalized_var"] = early_stopper(logs['eval_loss'])
                            if args.triple:
                                for interval in rand_perms_triple:
                                    logs[f"eval_loss_{interval}"] = triple_loss[interval] / args.eval_gradient_accumulation_steps

                progress_bar.update(1)
                if args.method == 'custom':
                    params_to_clip = (
                        itertools.chain([x[1] for x in unet.named_parameters() if ('attn2' in x[0])],
                                        text_encoder.parameters())
                    )
                    accelerator.clip_grad_norm_(params_to_clip, 1.)

                if args.optimizer == "sam":
                    optimizer.step(closure)
                else:
                    optimizer.step()
                lr_scheduler.step()

                global_step += 1
                if args.sample_frequency > 0 and global_step % args.sample_frequency == 0:
                    clip_scores = evaluate(pipeline, train_prompts, val_prompts, clip, reference_images, logging_dir,
                                           args.placeholder_token, logger=stat_logger, sample_steps=args.sample_steps,
                                           guidance=args.guidance, fp16=args.fp16, sampling_seed=args.sampling_seed,
                                           log_unscaled=args.log_unscaled, step=global_step)
                    wandb.log(clip_scores, commit=False, step=global_step)
                    if args.variant == "clip_early_stopping":
                        early_stopper(clip_scores['train_clip_img_score'])

            learned_embed = text_encoder.get_input_embeddings().concept_token.data

            if global_step % args.save_steps == 0:
                if accelerator.is_main_process:
                    name = f"learned_weights_{global_step:04d}.bin"
                    save_progress(object_to_save, args.placeholder_token, logging_dir, name=name, method=args.method)

            logs["lr"] = lr_scheduler.get_last_lr()[0]
            logs["emb_reg_loss"] = torch.norm(original_embed - learned_embed, 2).detach()

            if args.logger == "tensorboard":
                for k in logs:
                    stat_logger.add_scalar(k, logs[k], global_step)
            if args.logger == "wandb":
                wandb.log(logs, step=global_step)

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        # save last trained embeddings
        name = f"learned_weights_last.bin"
        save_progress(object_to_save, args.placeholder_token, logging_dir, name=name, method=args.method)

    accelerator.end_training()
    train_end = time.time()
    wandb.run.summary["train_time"] = train_end - train_start

    if args.variant in ['dvar_early_stopping', "short_iters"]:
        clip_scores = evaluate(pipeline, train_prompts, val_prompts, clip, reference_images, logging_dir,
                               args.placeholder_token, step=global_step, guidance=args.guidance,
                               sample_steps=args.sample_steps, sampling_seed=args.sampling_seed, fp16=args.fp16,
                               log_unscaled=args.log_unscaled)
        wandb.log(clip_scores)

        
if __name__ == "__main__":
    main()
