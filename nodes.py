"""ComfyUI nodes for loading and auditing Qwen-Image 2.1 LoRAs."""

import logging

import comfy.lora
import comfy.lora_convert
import comfy.utils
import folder_paths

from .gate_up import remap_lora
from .lora_keys import model_layout
from .report import build_report


def load_lora_file(path):
    """Load a LoRA file and normalize its key names."""
    lora = comfy.utils.load_torch_file(path, safe_load=True)
    return comfy.lora_convert.convert_lora(lora)


def build_key_maps(model, clip):
    """Build the model-side lookup maps for the inputs that are connected."""
    unet_map = comfy.lora.model_lora_keys_unet(model.model, {}) if model is not None else {}
    clip_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, {}) if clip is not None else {}
    return unet_map, clip_map


class Qwen21LoraAutoFix:
    """Load a LoRA and repair Qwen-Image 2.1 gate_up key mapping (MODEL + CLIP)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
            },
            "optional": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "strict": ("BOOLEAN", {"default": False,
                                       "tooltip": "True: 有模块无法映射就报错, 而不是跳过"}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "report")
    FUNCTION = "apply"
    CATEGORY = "Qwen-Image 2.1"
    DESCRIPTION = ("Qwen-Image 2.1 LoRA 加载器: 自动把 fused gate_up 无损拆成 gate_layer/proj, "
                   "适配拆分布局的 GGUF; 同时支持 MODEL 与 CLIP")

    def apply(self, lora_name, strength_model, strength_clip, model=None, clip=None, strict=False):
        if model is None and clip is None:
            raise RuntimeError("Qwen21LoraAutoFix: model 和 clip 至少接一个")

        path = folder_paths.get_full_path("loras", lora_name)
        lora = load_lora_file(path)
        unet_map, clip_map = build_key_maps(model, clip)
        key_map = {**unet_map, **clip_map}

        remapped, report = remap_lora(lora, unet_map, clip_map)
        if strict and report["unmatched"]:
            raise RuntimeError("LoRA 有 %d 个模块无法映射:\n%s"
                               % (len(report["unmatched"]), "\n".join(report["unmatched"][:10])))

        loaded = comfy.lora.load_lora(remapped, key_map)

        layout = model_layout(unet_map) if model is not None else "未接入 MODEL"
        new_model, new_clip = model, clip
        model_patches = clip_patches = None

        if model is not None:
            new_model = model.clone()
            model_patches = len(set(new_model.add_patches(loaded, strength_model)))
        if clip is not None:
            new_clip = clip.clone()
            clip_patches = len(set(new_clip.add_patches(loaded, strength_clip)))

        text = build_report("Qwen21LoraAutoFix", path, report, layout,
                            clip is not None, model_patches, clip_patches, len(loaded))

        if loaded and (model_patches or 0) + (clip_patches or 0) == 0:
            logging.warning("[Qwen21LoraAutoFix] 没有任何 patch 命中, 检查 LoRA 与模型是否匹配")

        return (new_model, new_clip, text)


class Qwen21LoraAudit:
    """Inspect LoRA key mapping without modifying the model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"),),
            },
            "optional": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "audit"
    CATEGORY = "Qwen-Image 2.1"
    OUTPUT_NODE = True
    DESCRIPTION = "检查一个 LoRA 能否完整映射到当前 MODEL / CLIP (含 GGUF)"

    def audit(self, lora_name, model=None, clip=None):
        if model is None and clip is None:
            raise RuntimeError("Qwen21LoraAudit: model 和 clip 至少接一个")

        path = folder_paths.get_full_path("loras", lora_name)
        lora = load_lora_file(path)
        unet_map, clip_map = build_key_maps(model, clip)
        _, report = remap_lora(lora, unet_map, clip_map)
        layout = model_layout(unet_map) if model is not None else "未接入 MODEL"

        text = build_report("Qwen21LoraAudit", path, report, layout, clip is not None, None, None, None)
        print(text)
        return {"ui": {"text": [text]}, "result": (text,)}


NODE_CLASS_MAPPINGS = {
    "Qwen21LoraAutoFix": Qwen21LoraAutoFix,
    "Qwen21LoraAudit": Qwen21LoraAudit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen21LoraAutoFix": "Qwen-Image 2.1 LoRA Loader (auto gate_up fix)",
    "Qwen21LoraAudit": "Qwen-Image 2.1 LoRA Audit",
}
