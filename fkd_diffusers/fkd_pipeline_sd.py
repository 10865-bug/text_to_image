# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import clip
import numpy as np
from PIL import Image


def _normalize_steering_type(s: Any) -> str:
    """Map TwoQuery aliases only. FK / URGE / FKG names are left unchanged."""
    raw = str(s)
    key = raw.upper().replace("-", "").replace("_", "")
    if key in ("TWOQUERY", "2Q", "TWOQ"):
        return "TWOQUERY"
    if key in ("SVDD", "SDVV"):
        return "SVDD"
    return raw

# Added for FK Steering
from fkd_diffusers.fkd_class import FKD, URGE, FKD_G
from fkd_diffusers.rewards import (
    get_reward_function,
    get_clip_guidance_gradient,
    REWARDS_DICT,
    clip_tensor_normalize,
    CLIPScore,
    do_clip_score,
    do_image_reward,
)

import torch
from packaging import version
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

# from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import LoraLoaderMixin
from diffusers.models import AutoencoderKL, ImageProjection, UNet2DConditionModel
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionPipeline,
    StableDiffusionPipelineOutput,
)


def postprocess_and_apply_reward_fn(*, decoded_images, prompts_b, guidance_reward_fn, device):
    """
    Unified reward wrapper: ensure rewards are 1D [K].
    """

    if guidance_reward_fn == "Clip-Score":
        r = torch.as_tensor(
            do_clip_score(images=decoded_images, prompts=prompts_b),
            device=device, dtype=torch.float32
        )

    elif guidance_reward_fn == "ImageReward":
        scores = do_image_reward(images=decoded_images, prompts=prompts_b)
        if not torch.is_tensor(scores):
            r = torch.as_tensor(scores, device=device, dtype=torch.float32)
        else:
            r = scores.to(device=device, dtype=torch.float32)

    else:
        raise ValueError(f"Unknown guidance_reward_fn: {guidance_reward_fn}")

    if r.dim() == 2 and r.shape[1] == 1:
        r = r.squeeze(1)
    elif r.dim() != 1:
        raise RuntimeError(f"[reward_fn] must return 1D [K], got {tuple(r.shape)}")

    return r


def slice_K(t: torch.Tensor, b: int, K: int) -> torch.Tensor:
    if t is None:
        return None
    if t.dim() >= 5 and t.shape[0] != K:
        return t[b]
    else:
        start = b * K
        end = start + K
        return t[start:end]

def make_clip_guidance_grad_fn(clip_model, clip_tensor_normalize, prompts_source_fn):
    """
    Returns a guidance_grad_fn(decoded_images) -> grad tensor in the same device/dtype.
    prompts_source_fn() must return list[str] of length K for current step (B*K).
    """
    def _grad_fn(decoded_images: torch.Tensor) -> torch.Tensor:
        # decoded_images: [K,3,H,W], assumed in [0,1]
        # Normalize for CLIP
        imgs_norm = clip_tensor_normalize(decoded_images)  # match CLIP preprocessing
        imgs_norm = imgs_norm.requires_grad_(True)

        # Get prompts for current batch (B*K)
        prompts_list = prompts_source_fn()
        text_tokens = clip.tokenize(prompts_list).to(imgs_norm.device)

        image_embeds = clip_model.encode_image(imgs_norm)
        text_embeds  = clip_model.encode_text(text_tokens)

        # Similarity; any scalar objective that increases alignment
        # Here we use cosine similarity per-sample, sum for batch objective
        # If your CLIPScore exposes a score API, prefer that.
        sim = torch.nn.functional.cosine_similarity(image_embeds, text_embeds, dim=1)  # [K]
        objective = sim.sum()

        # Backprop to images
        grads = torch.autograd.grad(objective, imgs_norm, retain_graph=False)[0]  # [K,3,H,W]

        # Map grads back to decoded_images dtype/device if needed
        grads = grads.to(decoded_images.dtype).to(decoded_images.device)

        # Optional: per-sample normalization (unit norm to avoid scale blow-up)
        gflat = grads.view(grads.size(0), -1)
        gnorm = gflat.norm(dim=1, keepdim=True) + 1e-8
        grads = grads / gnorm.view(-1, 1, 1, 1)

        return grads
    return _grad_fn

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def safe_scheduler_step(
    *,
    latents: torch.Tensor,            # [N, C, H, W]
    noise_pred: torch.Tensor,         # [N, C, H, W]
    t: torch.Tensor,                  # scalar timestep tensor (e.g., torch.tensor(t_i))
    scheduler,                        # diffusers scheduler instance (DDIM/PNDM/etc.)
    eta: float = 0.0,                 # DDIM stochasticity; only used by DDIMScheduler
    generator: torch.Generator = None,
    return_dict: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    A safe wrapper around diffusers Scheduler.step that:
      - uses the scheduler's own update rule for the given timestep `t`
      - returns prev_sample and (if provided) pred_original_sample
      - synthesizes sampled_noise when eta>0 under DDIM; otherwise zeros

    This avoids manual alpha/sigma arithmetic and index mismatches.
    """

    # Some schedulers (e.g., DDIMScheduler) support an `eta` attribute influencing stochasticity.
    if hasattr(scheduler, "eta"):
        try:
            scheduler.eta = float(eta)
        except Exception:
            pass

    # Call the official step; this handles prediction_type, alpha/sigma, and timestep semantics.
    step_out = scheduler.step(
        model_output=noise_pred,
        timestep=t,
        sample=latents,
        eta=float(eta),
        generator=generator,
        return_dict=True,
    )

    prev_sample = step_out.prev_sample

    # pred_original_sample is present for DDIM/DDPM-style schedulers when return_dict=True
    if hasattr(step_out, "pred_original_sample") and (step_out.pred_original_sample is not None):
        x0_pred = step_out.pred_original_sample
    else:
        # Not all schedulers return x0; provide a placeholder for interface consistency
        x0_pred = torch.zeros_like(prev_sample)

    # Sampled noise is not generally returned by diffusers; synthesize when eta>0 under DDIM
    if (hasattr(scheduler, "eta") and float(scheduler.eta) > 0.0):
        if generator is not None:
            sampled_noise = torch.randn_like(latents, generator=generator)
        else:
            sampled_noise = torch.randn_like(latents)
    else:
        sampled_noise = torch.zeros_like(latents)

    return {
        "prev_sample": prev_sample,
        "pred_original_sample": x0_pred,
        "sampled_noise": sampled_noise,
    }


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class FKDStableDiffusion(
    StableDiffusionPipeline,
    DiffusionPipeline,
    StableDiffusionMixin,
):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.StableDiffusionLoraLoaderMixin.save_lora_weights`] for saving LoRA weights
        - [`~loaders.FromSingleFileMixin.from_single_file`] for loading `.ckpt` files
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] for loading IP Adapters

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """

    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
    _exclude_from_cpu_offload = ["safety_checker"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            requires_safety_checker=requires_safety_checker,
        )
        
    def guidance_anneal(
        self,
        step_idx: int,
        num_steps: int,
        base_alpha: float,
        p: float = 2.0,
        min_scale: float = 0.2,
    ):
        """
        Quadratic annealing schedule:
        - Starts from base_alpha at step 0
        - Decreases quadratically with normalized step
        - Floors at min_scale * base_alpha
        """
        t = step_idx / max(1, num_steps)
        scale = (1.0 - t**p)
        scale = max(min_scale, scale)
        return base_alpha * scale
    
    def _encode_to_latents(self, x_img: torch.Tensor) -> torch.Tensor:
        assert x_img.dim() == 4 and x_img.shape[1] == 3, f"Expect RGB image, got {tuple(x_img.shape)}"
        x_img = x_img.to(device=self._execution_device, dtype=self.vae.dtype).contiguous()
        x_img = x_img.clamp(-1.0, 1.0)
        with torch.no_grad():
            posterior = self.vae.encode(x_img).latent_dist
            z = posterior.sample().contiguous() 
            z = self.vae.post_quant_conv(z)
            if hasattr(self.vae, "config") and hasattr(self.vae.config, "scaling_factor"):
                z = z * self.vae.config.scaling_factor
        return z

    def _decode_to_image(self, z: torch.Tensor) -> torch.Tensor:
        assert z.dim() == 4 and z.shape[1] == 4, f"Expect latent with 4 channels, got {tuple(z.shape)}"
        z = z.to(device=self._execution_device, dtype=self.vae.dtype).contiguous()
        if hasattr(self.vae, "config") and hasattr(self.vae.config, "scaling_factor"):
            z = z / self.vae.config.scaling_factor
        with torch.no_grad():
            x_img = self.vae.decode(z).sample 
        return x_img

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        fkd_args: Optional[Dict[str, Any]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 100,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        guidance_reward_fn: str = "ImageReward",
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            
            FK Steering Addition:
            fkd_args (`dict`, *optional*):
                The arguments to be passed to the FKD class. If not defined, FKD will not be used.

            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should
                contain the negative image embedding if `do_classifier_free_guidance` is set to `True`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
            
        if callable(callback_on_step_end):
            callback_on_step_end_tensor_inputs = ["latents"]

        # 0. Default resolution based on UNet sample_size and VAE scale factor
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Input validation
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        # Record minimal pipeline state (persisted for downstream use)
        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False
        self.num_inference_steps = int(num_inference_steps)
        self._guidance_reward_fn = str(guidance_reward_fn or "ImageReward")
        self._sigma_provider_fn = lambda idx: 1.0
        self._dt_provider_fn = lambda idx: 1.0 / max(1, getattr(self, "num_inference_steps", 50))
        self._fkd = None

        # 2. Determine batch size (B) from prompt or embeddings
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, (list, tuple)):
            batch_size = len(prompt)
        else:
            if prompt_embeds is None:
                raise RuntimeError("Either `prompt` or `prompt_embeds` must be provided.")
            batch_size = int(prompt_embeds.shape[0])

        # Device for execution
        device = self._execution_device

        # 3. Encode input prompt to embeddings
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None)
            if self.cross_attention_kwargs is not None
            else None
        )

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # Concatenate unconditional and text embeddings here (matches original pipeline)
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # Keep batch_size consistent with embeddings if prompt was None
        batch_size = int(prompt_embeds.shape[0] // (2 if self.do_classifier_free_guidance else 1))

        # Sanity check for num_images_per_prompt
        if (not isinstance(num_images_per_prompt, int)) or (num_images_per_prompt < 1):
            raise ValueError("`num_images_per_prompt` must be a positive integer.")

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 5. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 5.1 Add image embeds for IP-Adapter
        image_embeds = None
        if (ip_adapter_image is not None) or (ip_adapter_image_embeds is not None):
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,  # use batch_size (not target_batch)
                self.do_classifier_free_guidance,
            )

        added_cond_kwargs = {"image_embeds": image_embeds} if (image_embeds is not None) else None

        # 5.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(
                batch_size * num_images_per_prompt  # consistent with IP-Adapter repeat
            )
            # Use prompt_embeds.dtype for stability (avoid latent dtype jitter)
            dtype_for_cond = prompt_embeds.dtype if prompt_embeds is not None else torch.float32
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=dtype_for_cond)

        # ---- Steering Dispatch (must NOT depend on time_cond_proj_dim) ----
        _fkd_args: Dict[str, Any] = fkd_args or {}
        steering_type: str = _normalize_steering_type(_fkd_args.get("steering_type", "FK"))
        use_smc: bool = bool(_fkd_args.get("use_smc", False))

        # Providers for sigma and dt (need timesteps/scheduler ready)
        def sigma_provider_fn(idx: int) -> float:
            # Prefer scheduler.sigmas when available; fallback to sqrt(1 - alpha_t) or 1.0
            if hasattr(self.scheduler, "sigmas") and isinstance(self.scheduler.sigmas, torch.Tensor):
                return float(self.scheduler.sigmas[idx])
            if hasattr(self.scheduler, "alphas_cumprod") and isinstance(self.scheduler.alphas_cumprod, torch.Tensor):
                alpha_t = float(self.scheduler.alphas_cumprod[idx])
                return float((max(0.0, 1.0 - alpha_t)) ** 0.5)
            return 1.0

        def dt_provider_fn(idx: int) -> float:
            T = max(1, len(timesteps))
            return 1.0 / T

        self._sigma_provider_fn = sigma_provider_fn
        self._dt_provider_fn = dt_provider_fn

        def _clean_args(d: Dict[str, Any]) -> Dict[str, Any]:
            # Remove keys that are pipeline-level only
            blacklist = {
                "use_smc", "steering_type", "guidance_reward_fn",
                "eta", "time_steps",
                "prompt", "output_dir", "metrics_to_compute",
                "twoq_k", "twoq_tau", "k", "tau", "svdd_m", "m",
            }
            return {k: v for k, v in d.items() if k not in blacklist}

        # Per-batch prompts context for FK/URGE reward alignment
        self._current_prompts_b: Optional[List[str]] = None

        # Initialize steering handler
        self._fkd = None

        if use_smc:
            fkd_args_clean = _clean_args(_fkd_args)

            def _ensure_tensor(x, device, dtype=torch.float32):
                # Ensure tensor input for reward/latent_to_decode_fn; keep it contiguous and on device
                if isinstance(x, torch.Tensor):
                    t = x
                else:
                    try:
                        t = torch.stack(list(x), dim=0)
                    except Exception as e:
                        raise TypeError(
                            f"latent_to_decode_fn expects a Tensor or an iterable of image tensors; got {type(x)}"
                        ) from e
                return t.to(device=device, dtype=dtype).contiguous()

            if steering_type == "FK":
                # Reward path: avoid PIL/list conversion; use output_type='pt' to keep tensors on device
                self._fkd = FKD(
                    time_steps=len(timesteps),
                    reward_fn=lambda x_imgs_tensor: postprocess_and_apply_reward_fn(
                        decoded_images=self.image_processor.postprocess(
                            _ensure_tensor(x_imgs_tensor, device=device, dtype=torch.float32),
                            output_type="pt",  # keep tensors; avoid implicit normalization to PIL
                        ),
                        prompts_b=self._current_prompts_b,
                        guidance_reward_fn=self._guidance_reward_fn,
                        device=device,
                    ),
                    latent_to_decode_fn=lambda x_imgs_tensor: _ensure_tensor(
                        x_imgs_tensor, device=device, dtype=torch.float32
                    ),
                    device=device,
                    **fkd_args_clean,
                )

            elif steering_type == "FKG":
                self._fkg_shift_cache: Dict[int, Dict[str, torch.Tensor]] = {}

                def proposal_shift_info_fn(step: int) -> Dict[str, torch.Tensor]:
                    return self._fkg_shift_cache.get(step, {})

                self._fkd = FKD_G(
                    sigma_provider_fn=sigma_provider_fn,
                    dt_provider_fn=dt_provider_fn,
                    proposal_shift_info_fn=proposal_shift_info_fn,
                    time_steps=len(timesteps),
                    reward_fn=lambda x_imgs_tensor: postprocess_and_apply_reward_fn(
                        decoded_images=self.image_processor.postprocess(
                            _ensure_tensor(x_imgs_tensor, device=device, dtype=torch.float32),
                            output_type="pt",  # keep tensors
                        ),
                        prompts_b=self._current_prompts_b,
                        guidance_reward_fn=self._guidance_reward_fn,
                        device=device,
                    ),
                    latent_to_decode_fn=lambda x_imgs_tensor: _ensure_tensor(
                        x_imgs_tensor, device=device, dtype=torch.float32
                    ),
                    device=device,
                    **fkd_args_clean,
                )

            elif steering_type == "URGE":
                eta_cfg = float(_fkd_args.get("eta", 1.0))
                urge_ctor_kwargs = dict(
                    num_particles=int(_fkd_args.get("num_particles", 1)),
                    adaptive_resampling=bool(_fkd_args.get("adaptive_resampling", True)),
                    resample_frequency=int(_fkd_args.get("resample_frequency", 1)),
                    resampling_t_start=int(_fkd_args.get("resampling_t_start", _fkd_args.get("resample_t_start", 0))),
                    resampling_t_end=int(_fkd_args.get("resampling_t_end", _fkd_args.get("resample_t_end", len(timesteps) - 1))),
                    time_steps=len(timesteps),
                    noise_sampler_fn=None,
                    sigma_provider_fn=sigma_provider_fn,
                    dt_provider_fn=dt_provider_fn,
                    device=device,
                    ess_threshold_ratio=float(_fkd_args.get("ess_threshold_ratio", 0.5)),
                    use_last_step_resample=bool(_fkd_args.get("use_last_step_resample", False)),
                    eta=eta_cfg,
                    replacement=bool(_fkd_args.get("replacement", True)),
                    debug=bool(_fkd_args.get("debug", False)),
                )
                self._fkd = URGE(
                    reward_fn=lambda x_imgs_tensor: postprocess_and_apply_reward_fn(
                        decoded_images=self.image_processor.postprocess(
                            _ensure_tensor(x_imgs_tensor, device=device, dtype=torch.float32),
                            output_type="pt",  # keep tensors
                        ),
                        prompts_b=self._current_prompts_b,
                        guidance_reward_fn=self._guidance_reward_fn,
                        device=device,
                    ),
                    latent_to_decode_fn=lambda x_imgs_tensor: _ensure_tensor(
                        x_imgs_tensor, device=device, dtype=torch.float32
                    ),
                    **urge_ctor_kwargs,
                )

            elif steering_type == "TWOQUERY":
                from fkd_diffusers.twoq_stein import TwoQueryStein

                tau = _fkd_args.get("twoq_tau", None)
                if tau is None and _fkd_args.get("lmbda") is not None:
                    tau = 1.0 / max(float(_fkd_args["lmbda"]), 1e-8)
                if tau is None:
                    tau = 0.5
                self._fkd = TwoQueryStein(
                    k=int(_fkd_args.get("twoq_k", _fkd_args.get("k", 2))),
                    tau=float(tau),
                    device=device,
                )

            elif steering_type == "SVDD":
                from fkd_diffusers.svdd_decode import SVDDSoftValue

                tau = _fkd_args.get("twoq_tau", None)
                if tau is None and _fkd_args.get("lmbda") is not None:
                    tau = 1.0 / max(float(_fkd_args["lmbda"]), 1e-8)
                if tau is None:
                    tau = 0.5
                self._fkd = SVDDSoftValue(
                    m=int(_fkd_args.get("svdd_m", _fkd_args.get("m", 4))),
                    tau=float(tau),
                    mh=bool(_fkd_args.get("svdd_mh", False)),
                    device=device,
                )

            elif steering_type in ("G", "Base"):
                raise ValueError("steering_type in {G, Base} should not be used with --use_smc; use FK/FKG/URGE/TWOQUERY/SVDD instead.")
            else:
                raise ValueError(f"Unknown steering_type: {steering_type}")


        # 6. Unified CLIPScore init (initialize once on pipeline device)
        if REWARDS_DICT.get("Clip-Score", None) is None:
            REWARDS_DICT["Clip-Score"] = CLIPScore(download_root=".", device=device)
        clip_model = REWARDS_DICT["Clip-Score"].clip_model

        # ---- Denoising loop bookkeeping ----
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        # Steering configuration
        steering_type = (
            _normalize_steering_type(fkd_args.get("steering_type", "FK"))
            if fkd_args is not None
            else None
        )
        use_smc = bool(fkd_args.get("use_smc", False)) if fkd_args is not None else False
        _twoq_eta = float(eta)
        if fkd_args is not None and fkd_args.get("eta") is not None:
            _twoq_eta = float(fkd_args["eta"])
        _twoq_reward_name = str(guidance_reward_fn)
        if fkd_args is not None and fkd_args.get("guidance_reward_fn"):
            _twoq_reward_name = str(fkd_args["guidance_reward_fn"])

        # Construct UNet conditional embeddings
        # CFG concatenation was already applied earlier; use prompt_embeds directly here.
        prompt_embeds_base = prompt_embeds
        added_cond_kwargs_base = added_cond_kwargs

        # Prepare initial latents
        num_channels_latents = self.unet.config.in_channels

        # Effective batch size (B) is number of prompts times num_images_per_prompt
        B = int(batch_size * num_images_per_prompt)

        K_init = int(fkd_args.get("num_particles", 1)) if (use_smc and fkd_args is not None) else 1
        if K_init <= 0:
            raise RuntimeError(f"num_particles must be > 0, got {K_init}")

        total_batch = B * K_init  # final batch in denoising loop

        latents = self.prepare_latents(
            total_batch,                # B * K
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,        # keep dtype consistent with embeddings
            device,
            generator,
            latents,
        )

        if int(latents.shape[0]) != total_batch:
            raise RuntimeError(
                f"[SMC init] expected total_batch={total_batch} (B={B}, K={K_init}, NIPP={num_images_per_prompt}), got {latents.shape[0]}"
            )

        # ---- Denoising loop ----
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # 1) Ensure latents are [B*K, C, H, W]; flatten if a 5D tensor is passed
                if latents.dim() == 5:
                    B1, K1, C, Hh, Ww = latents.shape
                    latents = latents.reshape(B1 * K1, C, Hh, Ww)
                else:
                    C, Hh, Ww = latents.shape[1], latents.shape[2], latents.shape[3]

                # 2) Derive K and sanity check against batch_size (B)
                total_batch = latents.shape[0]
                if total_batch % (batch_size * num_images_per_prompt) != 0:
                    raise RuntimeError(
                        f"Inconsistent total_batch={total_batch} and B={batch_size * num_images_per_prompt}. "
                        f"Expect total_batch = B * K."
                    )
                B = int(batch_size * num_images_per_prompt)
                K = total_batch // B

                # SMC sanity: when using SMC, K must equal num_particles
                if use_smc and (fkd_args is not None) and ("num_particles" in fkd_args):
                    expected_K = int(fkd_args["num_particles"])
                    if K != expected_K:
                        raise RuntimeError(
                            f"[SMC sanity] step={i}: derived K={K} from latents vs num_particles={expected_K}. "
                            f"total_batch={total_batch}, B={B}"
                        )
                # 3) Reshape to [B, K, C, H, W] for per-b processing and build UNet input
                latents_bk = latents.view(B, K, C, Hh, Ww)  # [B, K, C, H, W]

                # Build UNet input; under CFG, duplicate latents to match encoder batch [2B*K, ...]
                if self.do_classifier_free_guidance:
                    latent_model_bk = torch.cat([latents_bk, latents_bk], dim=0)   # [2B, K, C, H, W]
                    latent_model_input = latent_model_bk.reshape(-1, C, Hh, Ww)    # [2B*K, C, H, W]
                else:
                    latent_model_input = latents_bk.reshape(-1, C, Hh, Ww)         # [B*K, C, H, W]

                # 4) Expand prompt embeddings per particle (repeat_interleave on K) to match UNet batch
                # prompt_embeds_base is [2B, ...] (CFG) or [B, ...] (no CFG)
                prompt_embeds_iter = prompt_embeds_base.repeat_interleave(K, dim=0)  # [2B*K, ...] or [B*K, ...]

                # 4.1) Expand timestep_cond to match encoder batch
                timestep_cond_iter = None
                if timestep_cond is not None:
                    base_len = prompt_embeds_base.shape[0]  # B or 2B
                    if timestep_cond.shape[0] == base_len:
                        timestep_cond_iter = timestep_cond.repeat_interleave(K, dim=0)  # [B*K] or [2B*K]
                    else:
                        timestep_cond_iter = timestep_cond
                    if timestep_cond_iter.shape[0] != prompt_embeds_iter.shape[0]:
                        raise RuntimeError(
                            f"timestep_cond_iter length mismatch: {timestep_cond_iter.shape[0]} "
                            f"vs prompt_embeds_iter {prompt_embeds_iter.shape[0]}"
                        )

                # 4.2) Expand added_cond_kwargs to match encoder batch
                added_cond_kwargs_iter = None
                if added_cond_kwargs_base is not None:
                    added_cond_kwargs_iter = {}
                    base_len = prompt_embeds_base.shape[0]  # B or 2B
                    for k_cond, v_cond in added_cond_kwargs_base.items():
                        if torch.is_tensor(v_cond) and v_cond.shape[0] == base_len:
                            added_cond_kwargs_iter[k_cond] = v_cond.repeat_interleave(K, dim=0)  # [B*K] or [2B*K]
                        else:
                            added_cond_kwargs_iter[k_cond] = v_cond

                # 5) UNet forward (scale input per scheduler)
                scaled_input = self.scheduler.scale_model_input(latent_model_input, t)
                noise_pred = self.unet(
                    scaled_input,
                    t,
                    encoder_hidden_states=prompt_embeds_iter,  # matches latent_model_input batch
                    timestep_cond=timestep_cond_iter,
                    cross_attention_kwargs=self._cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs_iter,
                    return_dict=False,
                )[0]

                # 6) CFG combine -> noise_pred becomes [B*K, C, H, W]
                if self.do_classifier_free_guidance:
                    if noise_pred.shape[0] % 2 != 0:
                        raise RuntimeError(
                            f"Invalid noise_pred batch for CFG: {noise_pred.shape[0]} not divisible by 2."
                        )
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2, dim=0)  # each [B*K, C, H, W]
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)


                # 7) Scheduler step with explicit noise output
                if latents.shape[0] != noise_pred.shape[0]:
                    raise RuntimeError(
                        f"Latents batch ({latents.shape[0]}) does not match noise_pred batch ({noise_pred.shape[0]})."
                    )

                if use_smc and steering_type == "URGE":
                    if self._fkd is None:
                        raise RuntimeError("URGE handler was not initialized")
                    from fkd_diffusers.urge_tweedie import apply_urge_tweedie_step

                    def _urge_reward(decoded_images, prompts_b):
                        return postprocess_and_apply_reward_fn(
                            decoded_images=decoded_images,
                            prompts_b=prompts_b,
                            guidance_reward_fn=guidance_reward_fn,
                            device=device,
                        )

                    latents = apply_urge_tweedie_step(
                        pipe=self,
                        latents=latents,
                        noise_pred=noise_pred,
                        t=t,
                        eta=float(eta),
                        prompt=prompt,
                        B=B,
                        K=K,
                        sampling_idx=i,
                        reward_fn=_urge_reward,
                        generator=generator,
                    )
                    if callable(callback_on_step_end):
                        callback_on_step_end(i, t, {"latents": latents})
                    progress_bar.update()
                    continue

                if use_smc and steering_type in ("TWOQUERY", "SVDD"):
                    if self._fkd is None:
                        raise RuntimeError(f"{steering_type} handler was not initialized")
                    if isinstance(prompt, list):
                        if len(prompt) == B:
                            prompts_list = [prompt[j // K] for j in range(B * K)]
                        elif len(prompt) == B * K:
                            prompts_list = prompt
                        else:
                            repeats = (B * K + len(prompt) - 1) // len(prompt)
                            prompts_list = (prompt * repeats)[: (B * K)]
                    else:
                        prompts_list = [prompt] * (B * K)

                    def _twoq_decode_x0(x0_latents: torch.Tensor) -> torch.Tensor:
                        vae_device = next(self.vae.parameters()).device
                        vae_dtype = getattr(self.vae, "dtype", next(self.vae.parameters()).dtype)
                        sf = float(self.vae.config.scaling_factor)
                        pieces = []
                        # M=16 with N=4 is 64 latents; one VAE batch of that size OOMs a 40GB GPU.
                        for start in range(0, int(x0_latents.shape[0]), 8):
                            z = x0_latents[start : start + 8].to(device=vae_device, dtype=vae_dtype)
                            decoded = self.vae.decode(z / sf).sample
                            decoded = decoded.to(device=device, dtype=torch.float32)
                            out = self.image_processor.postprocess(decoded, output_type="pt")
                            if not torch.is_tensor(out):
                                tensors = []
                                for im in out:
                                    t = torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float().div(255.0)
                                    tensors.append(t)
                                out = torch.stack(tensors, dim=0)
                            pieces.append(out.to(device=device, dtype=torch.float32))
                        return torch.cat(pieces, dim=0)

                    def _twoq_reward(decoded_images, prompts_b):
                        return postprocess_and_apply_reward_fn(
                            decoded_images=decoded_images,
                            prompts_b=prompts_b,
                            guidance_reward_fn=_twoq_reward_name,
                            device=device,
                        )

                    latents = self._fkd.step(
                        latents=latents,
                        noise_pred=noise_pred,
                        t=t,
                        scheduler=self.scheduler,
                        eta=_twoq_eta,
                        generator=generator,
                        prompts=prompts_list,
                        decode_x0_fn=_twoq_decode_x0,
                        reward_fn=_twoq_reward,
                    )
                    if callable(callback_on_step_end):
                        callback_on_step_end(i, t, {"latents": latents})
                    progress_bar.update()
                    continue

                step_dict = safe_scheduler_step(
                    latents=latents,
                    noise_pred=noise_pred,
                    t=t,
                    scheduler=self.scheduler,
                    eta=float(eta),
                    generator=generator,
                    return_dict=True,
                )

                latents = step_dict["prev_sample"]
                x0_preds = step_dict["pred_original_sample"]
                sampled_noise = step_dict["sampled_noise"]

                # 8) Prepare decoded x0_images in pixel domain (for reward/grad)
                V_t = self._sigma_provider_fn(i) if self._sigma_provider_fn else 1.0
                dt  = self._dt_provider_fn(i) if self._dt_provider_fn else 1.0 / max(1, len(timesteps))

                # Decode latent-space x0 to pixel-space images only if not already in [*,3,*,*]
                if isinstance(x0_preds, torch.Tensor) and x0_preds.dim() == 4 and x0_preds.shape[1] == 3:
                    x0_images = x0_preds.to(device=latents.device, dtype=torch.float32)
                else:
                    x0_latents = step_dict["pred_original_sample"]  # [B*K, C_lat, H_lat, W_lat]
                    vae_device = next(self.vae.parameters()).device
                    vae_dtype  = getattr(self.vae, "dtype", next(self.vae.parameters()).dtype)
                    x0_latents = x0_latents.to(device=vae_device, dtype=vae_dtype)
                    with torch.no_grad():
                        sf = torch.tensor(float(self.vae.config.scaling_factor), device=vae_device, dtype=vae_dtype)
                        decoded = self.vae.decode(x0_latents / sf).sample  # [B*K,3,H_img,W_img]
                    x0_images = decoded.to(device=latents.device, dtype=torch.float32)

                # Ensure pixel-space structure for downstream usage
                if x0_images.dim() != 4 or x0_images.shape[1] != 3:
                    raise RuntimeError(f"x0_images must be [B*K, 3, H_img, W_img], got {tuple(x0_images.shape)}")

                # 9) Build prompts_list with length B*K; support list[str] or single str
                if isinstance(prompt, list):
                    if len(prompt) == B:
                        prompts_list = [prompt[j // K] for j in range(B * K)]
                    elif len(prompt) == B * K:
                        prompts_list = prompt
                    else:
                        repeats = (B * K + len(prompt) - 1) // len(prompt)
                        prompts_list = (prompt * repeats)[: (B * K)]
                else:
                    prompts_list = [prompt] * (B * K)

                # === Per-b particle loop (compatible with FK / FKG / URGE; pass strict 4D [K,C,H,W]) ===
                K_global = int(fkd_args["num_particles"]) if (fkd_args is not None and "num_particles" in fkd_args) else K
                if K_global != K:
                    raise RuntimeError(f"[per-b loop] K_global ({K_global}) != derived K ({K}).")

                if latents.dim() != 4:
                    raise RuntimeError(f"latents must be [B*K, C_lat, H_lat, W_lat], got {tuple(latents.shape)}")
                if x0_images is None or x0_images.dim() != 4 or x0_images.shape[1] != 3:
                    raise RuntimeError(f"x0_images must be [B*K, 3, H_img, W_img], got {None if x0_images is None else tuple(x0_images.shape)}")

                C_lat, H_lat, W_lat = int(latents.shape[1]), int(latents.shape[2]), int(latents.shape[3])
                C_img, H_img, W_img = int(x0_images.shape[1]), int(x0_images.shape[2]), int(x0_images.shape[3])

                # Single view for per-b access
                latents_bk   = latents.view(B, K_global, C_lat, H_lat, W_lat)     # [B, K, C_lat, H_lat, W_lat]
                x0_images_bk = x0_images.view(B, K_global, C_img, H_img, W_img)   # [B, K, 3, H_img, W_img]

                # Precompute rewards for SMC (optional); keep tensor-only path to avoid CPU/PIL overhead
                if use_smc and (self._fkd is not None):
                    x0_images_fp32 = x0_images.to(device=device, dtype=torch.float32).contiguous()
                    r_all = postprocess_and_apply_reward_fn(
                        decoded_images=x0_images_fp32,
                        prompts_b=prompts_list,
                        guidance_reward_fn=guidance_reward_fn,  # e.g., "ImageReward"
                        device=device,
                    )
                    del x0_images_fp32
                else:
                    r_all = None

                # ---- Single per-batch dispatch loop (no duplication) ----
                for b_idx in range(B):
                    start = b_idx * K_global
                    end   = (b_idx + 1) * K_global

                    lat_k      = latents_bk[b_idx].contiguous()            # [K, C_lat, H_lat, W_lat]
                    img_k      = x0_images_bk[b_idx].contiguous()          # [K, 3, H_img, W_img]
                    prompts_k  = prompts_list[start:end]                    # length K
                    rewards_k  = r_all[start:end].contiguous() if r_all is not None else None
                    noise_k    = sampled_noise[start:end].contiguous()
                    x0_preds_k = x0_preds[start:end].contiguous()          # [K, 3, H_img, W_img] pixel-space

                    # ---- Non-SMC pixel guidance (G) ----
                    if (not use_smc) and (steering_type == "G"):
                        # Compute CLIP pixel-space gradient; keep tensor-only, no postprocess
                        x0_det = img_k.detach().to(device=device, dtype=torch.float32).requires_grad_(True)
                        with torch.enable_grad():
                            guidance_grad = get_clip_guidance_gradient(
                                decoded_images=x0_det,
                                prompts=prompts_k,
                                clip_model_bundle=(REWARDS_DICT["Clip-Score"].clip_model, clip_tensor_normalize),
                            )

                        # Use annealed alpha_g
                        base_alpha_g = float(fkd_args.get("g_alpha", 40.0))
                        alpha_g = self.guidance_anneal(
                            step_idx=i,
                            num_steps=num_inference_steps,
                            base_alpha=base_alpha_g,
                            p=float(fkd_args.get("g_anneal_p", 2.0)),
                            min_scale=float(fkd_args.get("g_min_scale", 0.5)), 
                        )
                        dt_local = dt

                        # Scheduler sigma at current step (DDIM has no sigmas, fallback to 1.0)
                        sigma_t = 1.0
                        if hasattr(self.scheduler, "sigmas"):
                            try:
                                idx = self.scheduler.index_for_timestep(int(t.item()))
                                sigma_t = float(self.scheduler.sigmas[idx])
                            except Exception:
                                sigma_t = 1.0

                        # Map pixel-space grad to latent resolution and channel count
                        lat_grad = torch.nn.functional.interpolate(
                            guidance_grad, size=(H_lat, W_lat), mode="bilinear", align_corners=False
                        )  # [K,3,H_lat,W_lat]
                        # Expand to latent channels (e.g., SD v1.x latent C=4)
                        lat_grad = torch.cat([lat_grad, lat_grad.mean(dim=1, keepdim=True)], dim=1)[:, :C_lat].contiguous()

                        # Apply guidance update with annealed alpha
                        delta_lat = (V_t ** 2) * alpha_g * dt_local * sigma_t * lat_grad
                        latents_bk[b_idx] = (lat_k + delta_lat).contiguous()

                        # ---- Debug: gradient and reward before/after ----
                        grad_norm = guidance_grad.norm().item()
                        print(f"[G Debug] step={i}, grad_norm={grad_norm:.4f}, alpha_g={alpha_g:.4f}, base_alpha={base_alpha_g}, sigma_t={sigma_t:.4f}")

                        with torch.no_grad():
                            vae_dev = next(self.vae.parameters()).device
                            vae_dt = getattr(self.vae, "dtype", next(self.vae.parameters()).dtype)
                            sf = float(self.vae.config.scaling_factor)
                            lat_pre = (lat_k.to(device=vae_dev, dtype=vae_dt) / sf).contiguous()
                            imgs_before = self.vae.decode(lat_pre).sample.to(
                                device=device, dtype=torch.float32
                            )
                            r_before = do_clip_score(images=imgs_before, prompts=prompts_k)

                            lat_post = ((lat_k + delta_lat).to(device=vae_dev, dtype=vae_dt) / sf).contiguous()
                            imgs_after = self.vae.decode(lat_post).sample.to(
                                device=device, dtype=torch.float32
                            )
                            r_after = do_clip_score(images=imgs_after, prompts=prompts_k)

                        print(f"[G Debug] step={i}, reward before={r_before.mean():.4f}, after={r_after.mean():.4f}")
                        continue
                    
                    # ---- SMC steering (FK / FKG / URGE) ----
                    latents_bk[b_idx] = lat_k.contiguous()

                    # ---- SMC steering branches ----
                    if use_smc and (self._fkd is not None):

                        if steering_type == "FK":
                            # Build per-b prompts context
                            if isinstance(prompt, list):
                                self._current_prompts_b = [prompt[b_idx]] * K_global
                            else:
                                self._current_prompts_b = [prompt] * K_global

                            x0_images_k = x0_images_bk[b_idx]  # [K,3,H_img,W_img]
                            assert x0_images_k.dim() == 4 and x0_images_k.shape[0] == K_global, \
                                f"[FK] x0_images_k shape invalid: {tuple(x0_images_k.shape)}"
                            assert x0_images_k.shape[1] == 3, \
                                f"[FK] expect RGB images, got C={x0_images_k.shape[1]}"

                            x0_images_k = x0_images_k.to(device=device, dtype=torch.float32).contiguous()
                            assert lat_k.shape == (K_global, C_lat, H_lat, W_lat), \
                                f"[FK] latent shape mismatch: got {tuple(lat_k.shape)}, expect {(K_global, C_lat, H_lat, W_lat)}"

                            # FK consumes pixel-space images directly
                            resampled_lat_k, _ = self._fkd.resample(
                                sampling_idx=i,
                                latents=lat_k,           # [K, C_lat, H_lat, W_lat]
                                x0_preds=x0_images_k,    # [K, 3, H_img, W_img]
                            )
                            if resampled_lat_k.shape != (K_global, C_lat, H_lat, W_lat):
                                resampled_lat_k = resampled_lat_k.view(K_global, C_lat, H_lat, W_lat)

                            latents_bk[b_idx] = resampled_lat_k.contiguous()

                        elif steering_type == "FKG":
                            # FK + G: apply pixel guidance displacement, then resample like FK using displaced images
                            # Inputs: lat_k [K,C_lat,H_lat,W_lat], img_k [K,3,H_img,W_img], prompts_k, dt, V_t
                            # Outputs: resampled latents after guidance displacement

                            # 1) Compute CLIP pixel-space gradient (tensor-only)
                            x0_det = img_k.detach().to(device=device, dtype=torch.float32).requires_grad_(True)  # [K,3,H_img,W_img]
                            with torch.enable_grad():
                                guidance_grad = get_clip_guidance_gradient(
                                    decoded_images=x0_det,
                                    prompts=prompts_k,
                                    clip_model_bundle=(REWARDS_DICT["Clip-Score"].clip_model, clip_tensor_normalize),
                                )
                            guidance_grad = guidance_grad.to(device=device, dtype=torch.float32).contiguous()

                            # 2) Annealed alpha_g (DDIM: no sigmas)
                            base_alpha_g = float(fkd_args.get("g_alpha", 40.0))
                            alpha_g = self.guidance_anneal(
                                step_idx=i,
                                num_steps=num_inference_steps,
                                base_alpha=base_alpha_g,
                                p=float(fkd_args.get("g_anneal_p", 2.0)),
                                min_scale=float(fkd_args.get("g_min_scale", 0.5)), 
                            )
                            dt_local = dt
                            sigma_t = 1.0

                            # 3) Map pixel-space gradient to latent resolution/channels (simple RGB+mean expansion)
                            lat_grad = torch.nn.functional.interpolate(
                                guidance_grad, size=(H_lat, W_lat), mode="bilinear", align_corners=False
                            )  # [K,3,H_lat,W_lat]
                            lat_grad = torch.cat([lat_grad, lat_grad.mean(dim=1, keepdim=True)], dim=1)[:, :C_lat].contiguous()  # [K,C_lat,H_lat,W_lat]

                            # 4) Guidance displacement in latent space
                            delta_lat = (V_t ** 2) * alpha_g * dt_local * sigma_t * lat_grad
                            lat_k_displaced = (lat_k + delta_lat).contiguous()

                            # 5) Decode displaced latents to pixel-space images for FK-style resampling
                            vae_device = next(self.vae.parameters()).device
                            vae_dtype  = getattr(self.vae, "dtype", next(self.vae.parameters()).dtype)
                            with torch.no_grad():
                                sf = torch.tensor(float(self.vae.config.scaling_factor), device=vae_device, dtype=vae_dtype)
                                decoded_steered = self.vae.decode((lat_k_displaced.to(vae_device, vae_dtype)) / sf).sample
                            x0_images_displaced = decoded_steered.to(device=device, dtype=torch.float32).contiguous()  # [K,3,H_img,W_img]

                            # 6) Resample using displaced latents and displaced images (FK-compatible path, no external rewards)
                            resampled_lat_k, _ = self._fkd.resample(
                                sampling_idx=i,
                                latents=lat_k_displaced,       # [K, C_lat, H_lat, W_lat]
                                x0_preds=x0_images_displaced,  # [K, 3, H_img, W_img]
                            )
                            if resampled_lat_k.shape != (K_global, C_lat, H_lat, W_lat):
                                resampled_lat_k = resampled_lat_k.view(K_global, C_lat, H_lat, W_lat)

                            latents_bk[b_idx] = resampled_lat_k.contiguous()

                        elif steering_type == "URGE":
                            # Current prompts context for this batch
                            if isinstance(prompt, list):
                                self._current_prompts_b = [prompt[b_idx]] * K_global
                            else:
                                self._current_prompts_b = [prompt] * K_global

                            # 1) External pixel-space guidance_grad and rewards r_t (tensor-only path)
                            x0_det = img_k.detach().to(device=device, dtype=torch.float32).requires_grad_(True)  # [K,3,H_img,W_img]
                            with torch.enable_grad():
                                guidance_grad = get_clip_guidance_gradient(
                                    decoded_images=x0_det,
                                    prompts=prompts_k,
                                    clip_model_bundle=(REWARDS_DICT["Clip-Score"].clip_model, clip_tensor_normalize),
                                ).detach()  # [K,3,H_img,W_img], float32

                            # Evaluate rewards directly on tensors; avoid postprocess/CPU round-trips
                            r_t = postprocess_and_apply_reward_fn(
                                decoded_images=x0_det.detach(),   # tensor [K,3,H_img,W_img]
                                prompts_b=prompts_k,
                                guidance_reward_fn=guidance_reward_fn,
                                device=device,
                            ).to(device=device, dtype=torch.float32).contiguous()  # [K]

                            # 2) V_t and dt from providers (fall back safely)
                            V_t_val = float(self._sigma_provider_fn(i)) if (self._sigma_provider_fn is not None) else 1.0
                            dt_val = float(self._dt_provider_fn(i)) if (self._dt_provider_fn is not None) else (1.0 / max(1, len(timesteps)))

                            # 3) URGE resample: do not pass rewards_next (class computes internal difference)
                            resampled_lat_k, indices = self._fkd.resample(
                                sampling_idx=i,
                                latents=lat_k,                    # [K, C_lat, H_lat, W_lat]
                                decoded_images=None,              # URGE does not use internal autograd here
                                rewards=r_t,                      # [K]
                                rewards_next=None,                # let class compute (r_t - population_rs)
                                guidance_grad=guidance_grad,      # [K,3,H_img,W_img]
                                V_t=V_t_val,
                                dt=dt_val,
                                noise_for_martingale=None,        # let URGE sample internally
                            )

                            # 4) Shape alignment and write-back
                            if resampled_lat_k.shape != (K_global, C_lat, H_lat, W_lat):
                                resampled_lat_k = resampled_lat_k.view(K_global, C_lat, H_lat, W_lat)
                            latents_bk[b_idx] = resampled_lat_k.contiguous()

                        elif steering_type in ("TWOQUERY", "SVDD"):
                            latents_bk[b_idx] = lat_k.contiguous()
                        else:
                            raise ValueError(f"Unknown steering_type under SMC: {steering_type}")

                    # ---- Flatten back to [B*K, C, H, W] for the next step ----
                    latents = latents_bk.reshape(B * K_global, C_lat, H_lat, W_lat).contiguous()



                # Flatten back to [B*K, C, H, W] for the next step
                latents = latents_bk.reshape(B * K_global, C_lat, H_lat, W_lat).contiguous()

                # Per-step callback and progress
                if callable(callback_on_step_end):
                    # Keep minimal payload to avoid unnecessary synchronization
                    callback_on_step_end(i, t, {"latents": latents})
                progress_bar.update()


        # ---- Final decode and postprocess ----
        if not output_type == "latent":
            # Move input to VAE device/dtype once; avoid repeated conversions
            vae_device = next(self.vae.parameters()).device
            vae_dtype  = getattr(self.vae, "dtype", next(self.vae.parameters()).dtype)
            decode_input = (latents.to(device=vae_device, dtype=vae_dtype) / self.vae.config.scaling_factor).contiguous()
            with torch.no_grad():
                image = self.vae.decode(
                    decode_input,
                    return_dict=False,
                    generator=generator,
                )[0]
            has_nsfw_concept = None
            del decode_input
        else:
            image = latents
            has_nsfw_concept = None

        # Optional cache cleanup; keep robust
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        # Postprocess only once at the end
        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize
        )

        # Offload models if hooks are enabled
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept
        )


def latent_to_decode(
    *,
    model,
    output_type: str,
    latents: torch.Tensor,
    max_decode_batch: int = 8,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Decode latents to pixel-space images, or return latents when output_type == 'latent'.

    - Accepts latents of shape [N, C, H, W] or [B, K, C, H, W].
    - Applies scaling_factor and optional latents_mean/std exactly as in the original pipeline.
    - Decodes in chunks to reduce peak memory.
    - Returns float32 images on the pipeline execution device (stable for downstream processing).
    """
    # If the caller wants latents, return as-is
    if output_type == "latent":
        return latents

    # Normalize shape to 4D [N, C, H, W]
    if latents.dim() == 5:
        B, K, C, H, W = latents.shape
        latents_4d = latents.view(B * K, C, H, W)
    elif latents.dim() == 4:
        latents_4d = latents
        B, K = None, None
    else:
        raise RuntimeError(f"Unexpected latents ndim {latents.dim()}, expected 4 or 5.")

    # Unscale / denormalize latents (original logic)
    has_latents_mean = hasattr(model.vae.config, "latents_mean") and model.vae.config.latents_mean is not None
    has_latents_std  = hasattr(model.vae.config, "latents_std")  and model.vae.config.latents_std  is not None

    vae_device = next(model.vae.parameters()).device
    vae_dtype  = next(model.vae.parameters()).dtype

    latents_4d = latents_4d.to(device=vae_device, dtype=vae_dtype)

    if has_latents_mean and has_latents_std:
        latents_mean = torch.tensor(model.vae.config.latents_mean, device=vae_device, dtype=vae_dtype).view(1, -1, 1, 1)
        latents_std  = torch.tensor(model.vae.config.latents_std,  device=vae_device, dtype=vae_dtype).view(1, -1, 1, 1)
        latents_4d   = latents_4d * latents_std / model.vae.config.scaling_factor + latents_mean
    else:
        latents_4d   = latents_4d / model.vae.config.scaling_factor

    # Chunked decode to control memory
    N = latents_4d.shape[0]
    bs = max(1, int(max_decode_batch))
    decoded_chunks = []

    for start in range(0, N, bs):
        end = min(start + bs, N)
        lat_chunk = latents_4d[start:end]
        img_chunk = model.vae.decode(lat_chunk, return_dict=False, generator=generator)[0]
        decoded_chunks.append(img_chunk)

    images = torch.cat(decoded_chunks, dim=0)

    # Return images as float32 on pipeline execution device (stable downstream)
    exec_device = getattr(model, "_execution_device", vae_device)
    images = images.to(device=exec_device, dtype=torch.float32)

    # Restore [B*K,...] shape if needed (optional, most callers expect flat N)
    # If you need to map back for later indexing, you can reshape here:
    # if B is not None and K is not None:
    #     images = images.view(B * K, images.shape[1], images.shape[2], images.shape[3])

    return images