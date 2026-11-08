"""LoRA key parsing and lookup helpers."""

SUFFIXES = [
    (".lora_A.default.weight", "A"),
    (".lora_B.default.weight", "B"),
    (".lora_A.weight", "A"),
    (".lora_B.weight", "B"),
    (".lora_down.weight", "A"),
    (".lora_up.weight", "B"),
    (".lora.down.weight", "A"),
    (".lora.up.weight", "B"),
    ("_lora.down.weight", "A"),
    ("_lora.up.weight", "B"),
    (".dora_scale", "D"),
    (".alpha", "alpha"),
]

PREFIXES = (
    "diffusion_model.",
    "model.diffusion_model.",
    "transformer.",
    "base_model.model.",
)

TE_PREFIX = "text_encoders."


def split_name(name):
    """Split a LoRA tensor name into (module base, part kind)."""
    for suffix, kind in SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)], kind
    return None, None


def normalize(base):
    """Strip known model prefixes until none remain."""
    changed = True
    while changed:
        changed = False
        for prefix in PREFIXES:
            if base.startswith(prefix):
                base = base[len(prefix):]
                changed = True
                break
    return base


def model_layout(unet_key_map):
    """Describe the model MLP layout using the keys it already exposes."""
    if any(".img_mlp.gate_up" in key for key in unet_key_map):
        return "gate_up (融合)"
    if any(".img_mlp.gate_layer" in key for key in unet_key_map):
        return "gate_layer + proj (拆分)"
    return "Not Qwen-Image 2.1 or unknown model struct"


def classify(base, unet_map, clip_map):
    """Return "unet", "clip", or None for a LoRA module base name."""
    normalized = normalize(base)
    for name in (base, normalized):
        if name in unet_map:
            return "unet"
        if name in clip_map:
            return "clip"
    if TE_PREFIX + normalized in clip_map:
        return "clip"
    if normalized.startswith(TE_PREFIX) and normalized[len(TE_PREFIX):] in clip_map:
        return "clip"
    return None
