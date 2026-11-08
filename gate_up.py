"""Remap fused gate_up LoRA blocks onto split gate_layer/proj weights."""

from .lora_keys import classify, normalize, split_name

GATE_UP = ".img_mlp.gate_up"
GATE_LAYER = ".img_mlp.gate_layer"
PROJ = ".img_mlp.proj"


def _group_entries(lora):
    """Group tensor names by module; names without a known suffix pass through."""
    groups = {}
    passthrough = {}
    for name, tensor in lora.items():
        base, kind = split_name(name)
        if base is None:
            passthrough[name] = tensor
        else:
            groups.setdefault(base, {})[kind] = name
    return groups, passthrough


def _copy_entries(out, kinds, lora):
    for name in kinds.values():
        out[name] = lora[name]


def _split_gate_up(base, kinds, lora):
    """Split one fused block. Returns (entries, error).

    B is cut by output rows; A and alpha are shared. Dora scales are cut only
    when they are per-row vectors matching the fused output size.
    """
    b_name = kinds.get("B")
    weight_b = lora.get(b_name) if b_name else None
    if weight_b is None:
        return None, "Missing lora_B, skipped"
    rows = weight_b.shape[0]
    if rows % 2:
        return None, "The row of lora_B is not even, skipped"

    half = rows // 2
    targets = (
        (base.replace(GATE_UP, GATE_LAYER), slice(0, half)),
        (base.replace(GATE_UP, PROJ), slice(half, None)),
    )
    out = {}
    for kind, name in kinds.items():
        tensor = lora[name]
        suffix = name[len(base):]
        for new_base, part in targets:
            if kind == "B" or (kind == "D" and tensor.dim() == 1 and tensor.shape[0] == rows):
                out[new_base + suffix] = tensor[part]
            else:
                out[new_base + suffix] = tensor
    return out, None


def remap_lora(lora, unet_map, clip_map):
    """Return (remapped lora, report).

    Keys that already resolve are kept as-is. Fused gate_up blocks are split
    only when the model exposes both gate_layer and proj. Everything else is
    passed through and recorded as unmatched for the caller to handle.
    """
    groups, passthrough = _group_entries(lora)
    out = dict(passthrough)
    report = {
        "total": len(groups),
        "unet": 0,
        "clip": 0,
        "split": [],
        "unmatched": [],
        "notes": [],
    }

    for base, kinds in groups.items():
        side = classify(base, unet_map, clip_map)
        if side is not None:
            report[side] += 1
            _copy_entries(out, kinds, lora)
            continue

        normalized = normalize(base)
        if normalized.endswith(GATE_UP):
            gate_base = normalized.replace(GATE_UP, GATE_LAYER)
            up_base = normalized.replace(GATE_UP, PROJ)
            if gate_base in unet_map and up_base in unet_map:
                entries, error = _split_gate_up(base, kinds, lora)
                if entries is None:
                    report["unmatched"].append(base)
                    report["notes"].append(f"{base}: {error}")
                    continue
                out.update(entries)
                report["split"].append(base)
                report["unet"] += 1
                continue

        report["unmatched"].append(base)
        _copy_entries(out, kinds, lora)

    return out, report
