import os
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

try:
    import hpsv2
except Exception:
    hpsv2 = None

# pip wheels of hpsv2 often omit this file; tokenizer import then crashes.
_HPS_BPE_NAME = "bpe_simple_vocab_16e6.txt.gz"
_HPS_BPE_URLS = (
    "https://raw.githubusercontent.com/mlfoundations/open_clip/main/src/open_clip/bpe_simple_vocab_16e6.txt.gz",
    "https://github.com/mlfoundations/open_clip/raw/main/src/open_clip/bpe_simple_vocab_16e6.txt.gz",
    "https://github.com/tgxs002/HPSv2/raw/master/hpsv2/src/open_clip/bpe_simple_vocab_16e6.txt.gz",
)


def hpsv2_bpe_path() -> str:
    if hpsv2 is None:
        raise ImportError("hpsv2 is not installed; HumanPreference scoring is unavailable.")
    return os.path.join(os.path.dirname(hpsv2.__file__), "src", "open_clip", _HPS_BPE_NAME)


def ensure_hpsv2_bpe() -> str:
    """Make sure hpsv2's bundled OpenCLIP BPE file exists before first score()."""
    dest = hpsv2_bpe_path()
    if os.path.isfile(dest) and os.path.getsize(dest) > 1000:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".tmp"
    last_err = None
    for url in _HPS_BPE_URLS:
        try:
            print(f"[hpsv2] downloading missing BPE from {url}")
            urllib.request.urlretrieve(url, tmp)
            with open(tmp, "rb") as handle:
                magic = handle.read(2)
            if magic != b"\x1f\x8b":
                raise RuntimeError(f"downloaded file is not gzip: {url}")
            if os.path.getsize(tmp) < 1000:
                raise RuntimeError(f"downloaded BPE is too small: {url}")
            os.replace(tmp, dest)
            print(f"[hpsv2] wrote {dest} ({os.path.getsize(dest)} bytes)")
            return dest
        except Exception as err:
            last_err = err
            if os.path.isfile(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
    raise FileNotFoundError(
        f"hpsv2 is missing {_HPS_BPE_NAME} at {dest}. "
        "Download that file into the path above and rerun."
    ) from last_err

try:
    from fkd_diffusers.llm_grading import LLMGrader
except Exception:
    LLMGrader = None

# Stores the reward models
REWARDS_DICT = {
    "Clip-Score": None,
    "ImageReward": None,
    "LLMGrader": None,
}


def _clip_eval_device() -> str:
    """Prefer CUDA when available; otherwise CPU (no hardcoded cuda)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def _clip_device_for_grad(decoded_images: torch.Tensor) -> str:
    """Gradient path must match the device of decoded latents/images."""
    if decoded_images.is_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _get_or_create_clip_score(*, device: str) -> "CLIPScore":
    """Recreate CLIP if missing or if cached model is on a different device."""
    global REWARDS_DICT
    cur = REWARDS_DICT.get("Clip-Score")
    if cur is None or getattr(cur, "device", None) != device:
        REWARDS_DICT["Clip-Score"] = CLIPScore(download_root=".", device=device)
    return REWARDS_DICT["Clip-Score"]


# Returns the reward function based on the guidance_reward_fn name
def get_reward_function(reward_name, images, prompts, metric_to_chase="overall_score"):
    #if reward_name != "LLMGrader":
        # print("`metric_to_chase` will be ignored as it only applies to 'LLMGrader' as the `reward_name`")
    if reward_name == "ImageReward":
        return do_image_reward(images=images, prompts=prompts)
    
    elif reward_name == "Clip-Score":
        return do_clip_score(images=images, prompts=prompts)
    
    elif reward_name == "HumanPreference":
        return do_human_preference_score(images=images, prompts=prompts)

    elif reward_name == "LLMGrader":
        return do_llm_grading(images=images, prompts=prompts, metric_to_chase=metric_to_chase)
    
    else:
        raise ValueError(f"Unknown metric: {reward_name}")
    

# Compute human preference score
def do_human_preference_score(*, images, prompts, use_paths=False):
    global hpsv2
    if hpsv2 is None:
        raise ImportError("hpsv2 is not installed; HumanPreference scoring is unavailable.")
    ensure_hpsv2_bpe()

    # Normalize images to list
    if isinstance(images, torch.Tensor):
        images_list = [images[i] for i in range(images.shape[0])]
    else:
        images_list = list(images)
    n_img = len(images_list)

    # Align prompts length to images length
    if isinstance(prompts, (list, tuple)):
        if len(prompts) == n_img:
            prompts_list = list(prompts)
        elif len(prompts) > 0:
            repeats = (n_img + len(prompts) - 1) // len(prompts)
            prompts_list = (list(prompts) * repeats)[:n_img]
        else:
            prompts_list = [""] * n_img
    else:
        prompts_list = [prompts] * n_img

    scores = []
    with torch.no_grad():
        if use_paths:
            raw_scores = hpsv2.score(images_list, prompts_list, hps_version="v2.1")
            scores = [float(s) for s in raw_scores]
        else:
            for i in range(n_img):
                score = hpsv2.score(images_list[i], prompts_list[i], hps_version="v2.1")
                if isinstance(score, (list, tuple)) or (hasattr(score, "shape") and score.shape):
                    score_val = float(score[0])
                else:
                    score_val = float(score)
                scores.append(score_val)

    return scores


def do_image_reward(*, images, prompts):
    global REWARDS_DICT
    if REWARDS_DICT.get("ImageReward", None) is None:
        from fkd_diffusers.image_reward_utils import rm_load

        REWARDS_DICT["ImageReward"] = rm_load("ImageReward-v1.0")

    if isinstance(images, torch.Tensor):
        images_list = [images[i] for i in range(images.shape[0])]
    else:
        images_list = list(images)
    n_img = len(images_list)

    if isinstance(prompts, (list, tuple)):
        if len(prompts) == n_img:
            prompts_list = list(prompts)
        elif len(prompts) > 0:
            repeats = (n_img + len(prompts) - 1) // len(prompts)
            prompts_list = (list(prompts) * repeats)[:n_img]
        else:
            prompts_list = [""] * n_img
    else:
        prompts_list = [prompts] * n_img

    with torch.no_grad():
        image_reward_result = REWARDS_DICT["ImageReward"].score_batched(prompts_list, images_list)

    return image_reward_result


# Compute CLIP-Score and diversity
def do_clip_score_diversity(*, images, prompts):
    """
    Robust CLIP-Score + diversity: align prompts/images lengths, flatten features,
    pairwise squared distance; uses CUDA if available else CPU.
    """
    clip_model = _get_or_create_clip_score(device=_clip_eval_device())

    if isinstance(images, torch.Tensor):
        images_list = [images[i] for i in range(images.shape[0])]
    else:
        images_list = list(images)

    n_img = len(images_list)

    if isinstance(prompts, (list, tuple)):
        if len(prompts) == n_img:
            prompts_list = list(prompts)
        elif len(prompts) > 0:
            repeats = (n_img + len(prompts) - 1) // len(prompts)
            prompts_list = (list(prompts) * repeats)[:n_img]
        else:
            prompts_list = [""] * n_img
    else:
        prompts_list = [prompts] * n_img

    if n_img == 0:
        return [], 0.0

    arr_clip_result = []
    img_feat_list = []

    with torch.no_grad():
        for i in range(n_img):
            prompt_i = prompts_list[i]
            img_i = images_list[i]
            clip_result, feature_vect = clip_model.score(
                prompt_i, img_i, return_feature=True
            )
            arr_clip_result.append(float(clip_result.item()))
            feat = feature_vect.get("image", None)
            if isinstance(feat, torch.Tensor):
                feat = feat.detach().float().cpu().view(-1)
            else:
                feat = torch.tensor(feat, dtype=torch.float32).view(-1)
            img_feat_list.append(feat)

    n_feat = len(img_feat_list)
    n = min(n_img, n_feat)
    if n <= 1:
        return arr_clip_result[:n], 0.0

    feats = torch.stack(img_feat_list[:n], dim=0)
    sq_norms = (feats ** 2).sum(dim=1, keepdim=True)
    gram = feats @ feats.t()
    dists = sq_norms + sq_norms.t() - 2.0 * gram
    dists = dists.clamp_min(0.0)
    off_diag_sum = dists.sum() - dists.diag().sum()
    diversity = float(off_diag_sum / (n * (n - 1)))

    return arr_clip_result[:n], diversity


def _clip_score_unwrap_image(x):
    """VAE decode may return DecoderOutput (.sample); list() on it is not a batch of tensors."""
    if isinstance(x, torch.Tensor):
        return x
    sample = getattr(x, "sample", None)
    if isinstance(sample, torch.Tensor):
        return sample
    return x


# Compute CLIP-Score
def do_clip_score(*, images, prompts):
    clip_model = _get_or_create_clip_score(device=_clip_eval_device())

    # DecoderOutput must unwrap before list(images): some versions iterate to string keys, not tensors.
    images = _clip_score_unwrap_image(images)
    if isinstance(images, torch.Tensor):
        images_list = [images[i] for i in range(images.shape[0])]
    else:
        images_list = [_clip_score_unwrap_image(x) for x in images]
    images_list = [_clip_score_unwrap_image(x) for x in images_list]
    n_img = len(images_list)

    if isinstance(prompts, (list, tuple)):
        if len(prompts) == n_img:
            prompts_list = list(prompts)
        elif len(prompts) > 0:
            repeats = (n_img + len(prompts) - 1) // len(prompts)
            prompts_list = (list(prompts) * repeats)[:n_img]
        else:
            prompts_list = [""] * n_img
    else:
        prompts_list = [prompts] * n_img

    scores = []
    for i in range(n_img):
        prompt = prompts_list[i]
        img = images_list[i]
        if isinstance(img, torch.Tensor):
            score = clip_model.score_tensor(prompt, img)
        else:
            score = clip_model.score(prompt, img)
            score = torch.tensor(score, device=clip_model.device, dtype=torch.float32)
        scores.append(score)
    scores = torch.stack(scores, dim=0).squeeze(-1)
    return scores


# Compute LLM-grading
def do_llm_grading(*, images, prompts, metric_to_chase="overall_score"):
    global REWARDS_DICT
    if LLMGrader is None:
        raise ImportError("LLMGrader is unavailable (google.generativeai not installed).")
    if REWARDS_DICT.get("LLMGrader", None) is None:
        REWARDS_DICT["LLMGrader"] = LLMGrader()

    # Normalize images to list
    if isinstance(images, torch.Tensor):
        images_list = [images[i] for i in range(images.shape[0])]
    else:
        images_list = list(images)
    n_img = len(images_list)

    # Align prompts length to images length
    if isinstance(prompts, (list, tuple)):
        if len(prompts) == n_img:
            prompts_list = list(prompts)
        elif len(prompts) > 0:
            repeats = (n_img + len(prompts) - 1) // len(prompts)
            prompts_list = (list(prompts) * repeats)[:n_img]
        else:
            prompts_list = [""] * n_img
    else:
        prompts_list = [prompts] * n_img

    llm_grading_result = []
    for i in range(n_img):
        result = REWARDS_DICT["LLMGrader"].score(
            images=images_list[i],
            prompts=prompts_list[i],
            metric_to_chase=metric_to_chase,
        )
        llm_grading_result.append(result)

    return llm_grading_result

class CLIPScore(nn.Module):
    def __init__(self, download_root, device='cpu'):
        super().__init__()
        self.device = device
        self.clip_model, self.preprocess = clip.load(
            "ViT-L/14", device=self.device, jit=False, download_root=download_root
        )

        if device == "cpu":
            self.clip_model.float()
        else:
            clip.model.convert_weights(
                self.clip_model
            )  # clip 

        # logit_scale
        self.clip_model.logit_scale.requires_grad_(False)
        self._mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                                  device=self.device).view(1, 3, 1, 1)
        self._std  = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                                  device=self.device).view(1, 3, 1, 1)

    def _tensor_normalize(self, tensor_image: torch.Tensor) -> torch.Tensor:
        if tensor_image.dim() == 3:
            tensor_image = tensor_image.unsqueeze(0)
        tensor_image = F.interpolate(tensor_image, size=(224,224),
                                     mode="bilinear", align_corners=False)
        return (tensor_image.to(self.device) - self._mean) / self._std

    def score(self, prompt, pil_image, return_feature=False):
        """
        PIL 
        """
        text = clip.tokenize(prompt, truncate=True).to(self.device)
        txt_features = F.normalize(self.clip_model.encode_text(text))
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        image_features = F.normalize(self.clip_model.encode_image(image))


        rewards = torch.sum(txt_features * image_features, dim=1, keepdim=True)

        if return_feature:
            return rewards, {'image': image_features, 'txt': txt_features}
        return rewards

    def score_tensor(self, prompt, tensor_image: torch.Tensor, return_feature=False):
        """
        Tensor 
        prompt: str
        tensor_image: torch.Tensor
        """
        text = clip.tokenize([prompt], truncate=True).to(self.device)
        txt_features = F.normalize(self.clip_model.encode_text(text), dim=-1)
        if not tensor_image.requires_grad:
            tensor_image = tensor_image.clone().detach().requires_grad_(True)

        image_norm = self._tensor_normalize(tensor_image)
        image_features = F.normalize(self.clip_model.encode_image(image_norm), dim=-1)

        rewards = torch.sum(txt_features * image_features, dim=1, keepdim=True)

        if return_feature:
            return rewards, {'image': image_features, 'txt': txt_features}
        return rewards


def clip_tensor_normalize(x: torch.Tensor) -> torch.Tensor:
    device = x.device
    x = F.interpolate(x, size=(224,224), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
    return (x - mean) / std


def get_clip_guidance_gradient(
    *,
    decoded_images: torch.Tensor,
    prompts: list,
    clip_model_bundle=None
) -> torch.Tensor:
    """
    decoded_images: [K,3,H,W], float tensor in [0,1]
    prompts: list[str], len K
    clip_model_bundle: (clip_model, norm_fn) optional
    returns: grad wrt decoded_images, [K,3,H,W]
    """
    global REWARDS_DICT

    if clip_model_bundle is None:
        dev = _clip_device_for_grad(decoded_images)
        clip_score = _get_or_create_clip_score(device=dev)
        clip_model = clip_score.clip_model
        norm_fn = clip_tensor_normalize
    else:
        clip_model, norm_fn = clip_model_bundle

    mdev = next(clip_model.parameters()).device
    text = clip.tokenize(prompts, truncate=True).to(mdev)
    txt_features = F.normalize(clip_model.encode_text(text), dim=-1)

    imgs = decoded_images.to(mdev).clone().detach().requires_grad_(True)
    imgs_norm = norm_fn(imgs)
    img_features = F.normalize(clip_model.encode_image(imgs_norm), dim=-1)

    scores = (txt_features * img_features).sum(dim=1)

    grad = torch.autograd.grad(
        outputs=scores.sum(),
        inputs=imgs,
        retain_graph=False,
        create_graph=False
    )[0]
    return grad
