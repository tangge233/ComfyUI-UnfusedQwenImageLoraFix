"""Report formatting for the Qwen-Image 2.1 LoRA nodes."""

import os


def build_report(title, path, report, layout, has_clip, model_patches,
                 clip_patches, total_patches):
    lines = [
        "[%s] %s" % (title, os.path.basename(path)),
        "  Model MLP layout : %s" % layout,
        "  LoRA modules     : %d" % report["total"],
        "    Diffusion side : %d" % report["unet"],
        "    Text encoder   : %d%s" % (
            report["clip"],
            "" if has_clip else "  (CLIP not connected, ignored)",
        ),
    ]

    if report["split"]:
        lines.append("  Lossless splits  : %d gate_up -> gate_layer + proj" % len(report["split"]))
        lines.append("                    e.g. %s" % report["split"][0])
    if report["unmatched"]:
        lines.append("  Unmatched        : %d modules" % len(report["unmatched"]))
        for name in report["unmatched"][:5]:
            lines.append("                    %s" % name)
    for note in report["notes"][:5]:
        lines.append("  Note             : %s" % note)
    if model_patches is not None:
        lines.append("  Model patches    : %d" % model_patches)
    if clip_patches is not None:
        lines.append("  Clip patches     : %d" % clip_patches)
    if total_patches is not None:
        lines.append("  Patch entries    : %d" % total_patches)
    if not report["unmatched"] and not report["split"]:
        lines.append("  Result           : native loading maps everything, no fix needed")
    return "\n".join(lines)
