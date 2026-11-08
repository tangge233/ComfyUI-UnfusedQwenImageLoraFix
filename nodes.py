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
                                       "tooltip": "True: Throw the error if mapping is failed"}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "report")
    FUNCTION = "apply"
    CATEGORY = "Qwen-Image 2.1"
    DESCRIPTION = ("Qwen-Image 2.1 LoRA Loader: split fused gate_up to gate_layer/proj, "
                   "Good for GGUF model; Supporting MODEL and CLIP jsut like the normal LoRA Loader")

    def apply(self, lora_name, strength_model, strength_clip, model=None, clip=None, strict=False):
        if model is None and clip is None:
            raise RuntimeError("Qwen21LoraAutoFix: no model or clip provided")

        path = folder_paths.get_full_path("loras", lora_name)
        lora = load_lora_file(path)
        unet_map, clip_map = build_key_maps(model, clip)
        key_map = {**unet_map, **clip_map}

        remapped, report = remap_lora(lora, unet_map, clip_map)
        if strict and report["unmatched"]:
            raise RuntimeError(f"Unable to mapping {len(report["unmatched"])} modules in LoRA:\n{"\n".join(report["unmatched"][:10])}")

        loaded = comfy.lora.load_lora(remapped, key_map)

        layout = model_layout(unet_map) if model is not None else "No MODEL provided"
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
            logging.warning("[Qwen21LoraAutoFix] No patch applied, please checking the LoRA file.")

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
    DESCRIPTION = "Check whether the LoRA can fully mapping to current MODEL/CLIP(may in GGUF) or not."

    def audit(self, lora_name, model=None, clip=None):
        if model is None and clip is None:
            raise RuntimeError("Qwen21LoraAutoFix: no model or clip provided")

        path = folder_paths.get_full_path("loras", lora_name)
        lora = load_lora_file(path)
        unet_map, clip_map = build_key_maps(model, clip)
        _, report = remap_lora(lora, unet_map, clip_map)
        layout = model_layout(unet_map) if model is not None else "No MODEL provided"

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
