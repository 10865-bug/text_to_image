from typing import Optional, Union
import os
import sys
import types

import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms.functional import pil_to_tensor

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _ensure_imagereward_without_refl() -> None:
    """ImageReward.__init__ imports ReFL, which imports all of diffusers. Stub it."""
    if "ImageReward.ReFL" not in sys.modules:
        sys.modules["ImageReward.ReFL"] = types.ModuleType("ImageReward.ReFL")


def _blip_pretrain_cls():
    _ensure_imagereward_without_refl()
    from ImageReward.models.BLIP.blip_pretrain import BLIP_Pretrain

    return BLIP_Pretrain


def _convert_image_to_rgb(image):
    if isinstance(image, torch.Tensor):
        from torchvision.transforms.functional import to_pil_image
        image = to_pil_image(image.cpu())
    return image.convert("RGB")


def _transform(n_px):
    return Compose(
        [
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size

        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            # nn.ReLU(),
            nn.Linear(16, 1),
        )

        # initial MLP param
        for name, param in self.layers.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=1.0 / (self.input_size + 1))
            if 'bias' in name:
                nn.init.constant_(param, val=0)

    def forward(self, input):
        return self.layers(input)

class IRSMC(nn.Module):
    def __init__(self, med_config, device='cpu'):
        super().__init__()
        self.device = device

        self.blip = _blip_pretrain_cls()(image_size=224, vit="large", med_config=med_config)
        self.preprocess = _transform(224)
        self.mlp = MLP(768)

        self.mean = 0.16717362830052426
        self.std = 1.0333394966054072

    def score_batched_old(self, prompts, images):
        # batch
        results = []
        for i, prompt in enumerate(prompts):
            results.append(self.score(prompt, images[i]))

        return results

    def score_gard(self, prompt_ids, prompt_attention_mask, image):
        image_embeds = self.blip.visual_encoder(image)
        # text encode cross attention with image
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            self.device
        )
        text_output = self.blip.text_encoder(
            prompt_ids,
            attention_mask=prompt_attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        txt_features = text_output.last_hidden_state[:, 0, :]  # (feature_dim)
        rewards = self.mlp(txt_features)
        rewards = (rewards - self.mean) / self.std

        return rewards

    def score(self, prompt, image):
        if type(image).__name__ == 'list':
            _, rewards = self.inference_rank(prompt, image)
            return rewards

        # text encode
        text_input = self.blip.tokenizer(
            prompt,
            padding='max_length',
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(self.device)

        # image encode
        if isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, str) and os.path.isfile(image):
            pil_image = Image.open(image)
        else:
            raise TypeError(
                r'This image parameter type has not been supportted yet. Please pass PIL.Image or file path str.'
            )

        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        image_embeds = self.blip.visual_encoder(image)

        # text encode cross attention with image
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            self.device
        )
        text_output = self.blip.text_encoder(
            text_input.input_ids,
            attention_mask=text_input.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        txt_features = text_output.last_hidden_state[:, 0, :].float()  # (feature_dim)
        rewards = self.mlp(txt_features)
        rewards = (rewards - self.mean) / self.std

        return rewards.detach().cpu().numpy().item()

    def score_batched(self, prompts, images):
        # Normalize prompts to List[str]
        if isinstance(prompts, torch.Tensor):
            prompts = [str(x) for x in prompts.tolist()]
        elif not isinstance(prompts, (list, tuple)):
            prompts = [str(prompts)]
        else:
            prompts = [str(x) for x in prompts]

        # Normalize images to list
        if isinstance(images, torch.Tensor):
            images = [images]
        elif not isinstance(images, (list, tuple)):
            images = [images]

        # text encode
        text_input = self.blip.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(self.device)

        # image encode
        images = [
            self.preprocess(image).unsqueeze(0).to(self.device) for image in images
        ]
        images = torch.cat(images, 0).to(self.device)

        image_embeds = self.blip.visual_encoder(images)

        # text encode cross attention with image
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            self.device
        )
        text_output = self.blip.text_encoder(
            text_input.input_ids,
            attention_mask=text_input.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        txt_features = text_output.last_hidden_state[:, 0, :].float()  # (feature_dim)
        rewards = self.mlp(txt_features)
        rewards = (rewards - self.mean) / self.std

        return rewards.view(txt_features.shape[0]).detach().cpu().numpy().tolist()

    def inference_rank(self, prompt, generations_list):
        text_input = self.blip.tokenizer(
            prompt,
            padding='max_length',
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(self.device)
        txt_set = []
        for generation in generations_list:
            # image encode
            if isinstance(generation, Image.Image):
                pil_image = generation
            elif isinstance(generation, str):
                if os.path.isfile(generation):
                    pil_image = Image.open(generation)
            else:
                raise TypeError(
                    r'This image parameter type has not been supportted yet. Please pass PIL.Image or file path str.'
                )
            image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            image_embeds = self.blip.visual_encoder(image)

            # text encode cross attention with image
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                self.device
            )
            text_output = self.blip.text_encoder(
                text_input.input_ids,
                attention_mask=text_input.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            txt_set.append(text_output.last_hidden_state[:, 0, :])

        txt_features = torch.cat(txt_set, 0).float()  # [image_num, feature_dim]
        rewards = self.mlp(txt_features)  # [image_num, 1]
        rewards = (rewards - self.mean) / self.std
        rewards = torch.squeeze(rewards)
        _, rank = torch.sort(rewards, dim=0, descending=True)
        _, indices = torch.sort(rank, dim=0)
        indices = indices + 1

        return (
            indices.detach().cpu().numpy().tolist(),
            rewards.detach().cpu().numpy().tolist(),
        )


def _hf_download(filename: str, download_root: str) -> str:
    from huggingface_hub import hf_hub_download

    os.makedirs(download_root, exist_ok=True)
    return hf_hub_download(
        repo_id="THUDM/ImageReward",
        filename=filename,
        local_dir=download_root,
    )


def rm_load(
    name: str = "ImageReward-v1.0",
    device: Optional[Union[str, torch.device]] = None,
    download_root: str = None,
    med_config: str = None,
):
    """Load ImageReward weights from Hugging Face. Does not import ImageReward.ReFL."""
    if device is None:
        device = "cpu"
    cache = download_root or os.path.expanduser("~/.cache/ImageReward")
    if os.path.isfile(name):
        model_path = name
    else:
        model_path = _hf_download("ImageReward.pt", cache)

    print("load checkpoint from %s" % model_path)
    state_dict = torch.load(model_path, map_location="cpu")

    if med_config is None:
        med_config = _hf_download("med_config.json", cache)

    model = IRSMC(device=device, med_config=med_config).to(device)
    model.load_state_dict(state_dict, strict=False)
    print("checkpoint loaded")
    model.eval()
    return model
