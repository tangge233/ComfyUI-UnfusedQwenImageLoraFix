"""Report formatting for the Qwen-Image 2.1 LoRA nodes."""

import os


def build_report(title, path, report, layout, has_clip, model_patches,
                 clip_patches, total_patches):
    lines = [
        "[%s] %s" % (title, os.path.basename(path)),
        "  模型 MLP 布局 : %s" % layout,
        "  LoRA 模块     : %d" % report["total"],
        "    扩散模型侧  : %d" % report["unet"],
        "    文本编码器侧: %d%s" % (
            report["clip"],
            "" if has_clip else "  (未接入 CLIP, 这部分会被忽略)",
        ),
    ]

    if report["split"]:
        lines.append("  无损拆分      : %d 个 gate_up -> gate_layer + proj" % len(report["split"]))
        lines.append("                  例: %s" % report["split"][0])
    if report["unmatched"]:
        lines.append("  未映射        : %d 个模块" % len(report["unmatched"]))
        for name in report["unmatched"][:5]:
            lines.append("                  %s" % name)
    for note in report["notes"][:5]:
        lines.append("  注意          : %s" % note)
    if model_patches is not None:
        lines.append("  model patch   : %d" % model_patches)
    if clip_patches is not None:
        lines.append("  clip patch    : %d" % clip_patches)
    if total_patches is not None:
        lines.append("  patch 条目    : %d" % total_patches)
    if not report["unmatched"] and not report["split"]:
        lines.append("  结果          : 原生即可完整映射, 无需处理")
    return "\n".join(lines)
