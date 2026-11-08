# ComfyUI-UnfusedQwenImageLoraFix

English | [中文](README.md)

A LoRA loader for Qwen-Image 2.1. Functionally it's the same as ComfyUI's built-in `LoraLoader`,
and it handles both the MODEL and CLIP paths. The difference is that it also fixes a key-name
mismatch: without the fix, some of the LoRA weights get silently dropped, and nothing errors out.

The node never touches LoRA files on disk and pulls in no extra dependencies — it uses only
ComfyUI's own loading and patching pipeline.

---

## What's the problem

Qwen-Image 2.1's SwiGLU layer can store its weights in two equivalent ways:

| Layout | Weight names | Shape |
|---|---|---|
| fused | `img_mlp.gate_up` | `[24576, 4096]`, first half is gate, second half is up |
| unfused | `img_mlp.gate_layer` + `img_mlp.proj` | `[12288, 4096]` each |

Mathematically these are the same thing; it's just a question of how the matrix is arranged.
The trouble is that the ecosystem has both: GGUF quantized models are almost always unfused,
while Qwen 2.1 LoRAs trained with kohya / sd-scripts are almost always fused.

ComfyUI core handles only one direction: when the model is fused and the LoRA is unfused, it
slices rows automatically. The other way around — unfused model, fused LoRA — the LoRA's
`gate_up` has no matching weight in the model, so the whole block gets skipped.

| Model | LoRA | Native loading result |
|---|---|---|
| fused | unfused | sliced automatically, fine |
| fused | fused | fine |
| unfused | unfused | fine |
| **unfused** | **fused** | **entire `gate_up` block dropped** |

That last row is not a small deal. A Qwen 2.1 LoRA usually has 224 modules, 32 of which are
`img_mlp.gate_up` — roughly 37% of the LoRA's total parameters. The console leaves a single
`lora key not loaded` line, nothing errors, and the result just looks like "the LoRA did
something, but not quite enough." That's the hardest kind of problem to track down, because
everything looks fine.

## How the node fixes it

At load time it splits `gate_up`'s `lora_B` into two halves by output row, with `lora_A` shared
by both:

```
delta_gate = B[:H] @ A        # B: [2H, r]
delta_up   = B[H:] @ A
delta_gate + delta_up == B @ A
```

The two halves concatenate back to exactly the original `B @ A`, so the rank is unchanged and
there is no approximation error. The split keys then go through ComfyUI's native `load_lora`,
so the rest of the patching pipeline is the same path the built-in node uses. The LoRA file
itself is never modified or rewritten.

---

## Installation

Drop the whole directory into ComfyUI's `custom_nodes`:

```bash
cp -r ComfyUI-UnfusedQwenImageLoraFix ~/ComfyUI/custom_nodes/
```

If you want repo updates to take effect directly, use a symlink:

```bash
ln -s "$(pwd)/ComfyUI-UnfusedQwenImageLoraFix" ~/ComfyUI/custom_nodes/ComfyUI-UnfusedQwenImageLoraFix
```

After restarting ComfyUI, the nodes show up under the **Qwen-Image 2.1** category. No extra
Python packages needed.

---

## Node 1: Qwen-Image 2.1 LoRA Loader (auto gate_up fix)

Internal id is `Qwen21LoraAutoFix`. This is the main node for everyday use.

### Inputs

| Name | Type | Description |
|---|---|---|
| `lora_name` | dropdown | a file from `models/loras` |
| `strength_model` | FLOAT | diffusion model strength, default 1.0 |
| `strength_clip` | FLOAT | text encoder strength, default 1.0 |
| `model` | MODEL (optional) | diffusion model; GGUF goes here too |
| `clip` | CLIP (optional) | text encoder; connect the output of `CLIPLoader` |
| `strict` | BOOLEAN | when on, any module that fails to map raises an error; off by default |

Connect at least one of `model` and `clip`. With only `model`, it works like
`LoraLoaderModelOnly`; with both, it works like the native `LoraLoader` and behaves the same way.

### Outputs

| Name | Type | Description |
|---|---|---|
| `model` | MODEL | patched model; returns `None` if not connected |
| `clip` | CLIP | patched CLIP; `None` if not connected |
| `report` | STRING | mapping report; worth wiring to a `Show Text` node |

### Reading the report

Loading a LoRA trained only on the diffusion model produces something like this:

```
[Qwen21LoraAutoFix] zzz_chibi_v2_st2500.safetensors
  Model MLP layout : gate_layer + proj (unfused)
  LoRA modules     : 192
    Diffusion side : 192
    Text encoder   : 0
  Lossless splits  : 32 gate_up -> gate_layer + proj
                    e.g. diffusion_model.transformer_blocks.0.img_mlp.gate_up
  Model patches    : 192
  Clip patches     : 0
  Patch entries    : 224
```

What the fields mean:

| Field | Meaning |
|---|---|
| `Model MLP layout` | `gate_up (fused)` means the model stores the weights fused; `gate_layer + proj (unfused)` means split |
| `Diffusion side` / `Text encoder` | how many modules in this LoRA belong to each side |
| `Lossless splits` | how many `gate_up` blocks were actually split; 0 means none needed splitting |
| `Unmatched` | modules that genuinely don't match; listed one by one |
| `Model patches` / `Clip patches` | how many patches landed on the model and the CLIP respectively |
| `Patch entries` | total number of patches after deduplication |

Normally **`Unmatched` should be 0**. If `Model patches` is much lower than `Diffusion side`,
the model has no matching weights — check whether you're using the wrong model version:
Qwen-Image 1.0 has 60 layers, 2.1 has 32, and their key names aren't interchangeable.

---

## Node 2: Qwen-Image 2.1 LoRA Audit

Internal id is `Qwen21LoraAudit`. It only inspects; it never modifies the model. The report is
shown both on the node and in the console. Before running a workflow you can use it to confirm
three things:

- whether this LoRA is fused or unfused;
- which layout the current GGUF model uses;
- roughly how much would be dropped if you loaded it natively.

Usage is simple: connect `MODEL` / `CLIP`, pick a `lora_name`, and wire `report` to `Show Text`.

---

## Typical wiring

**Diffusion model only** (the most common case; the vast majority of Qwen-Image LoRAs are like this)

```
Unet Loader (GGUF) ──┐
                     ├─ Qwen21LoraAutoFix ─ model ─ KSampler
CLIPLoader ──────────┘   (clip not connected)
```

**MODEL + CLIP** (when the LoRA carries text encoder weights)

```
Unet Loader (GGUF) ── model ─┐
                             ├─ Qwen21LoraAutoFix ─ model ─ KSampler
CLIPLoader ────────── clip ──┘        └──── clip ── CLIP Text Encode
```

**Stacking multiple LoRAs**

Just chain multiple `Qwen21LoraAutoFix` nodes, each with its own `strength_model`. This is no
different from chaining native `LoraLoader` nodes; patches accumulate normally.

---

## FAQ

### Can it directly replace the native LoraLoader?

Yes. Internally it goes through `comfy.lora.model_lora_keys_unet`, `model_lora_keys_clip`,
`load_lora`, and `add_patches`, with one extra key-name normalization step in between. If your
model and LoRA already match perfectly, the native node is completely fine too.

### Why not just repack the LoRA file?

That works too — that's what `lora_repack.py split` does: modify the file offline so any
frontend can use it. The node's advantage is that it leaves the original file alone, makes
swapping LoRAs and strengths easy, and doesn't leave you with a duplicate copy taking up space.

### CLIP is connected, but `Text encoder` is 0?

That means the LoRA was trained on the diffusion model only. That's true for the vast majority
of Qwen-Image LoRAs. Connecting CLIP won't affect the result; `strength_clip` just has no
effect.

### When should I turn on strict?

When tuning parameters or trying a new LoRA. For example, if you want to confirm that a LoRA
matches the current model exactly, turn on `strict` and the error will list every mismatched
module name. For production workflows, leave it off so one odd module doesn't fail the whole
run.

### My model is already fused — do I still need this node?

No. ComfyUI core already handles a fused model with an unfused LoRA, and splitting again would
be wrong. You can check `Model MLP layout` with `Qwen21LoraAudit` first; if it says
`gate_up (fused)`, just use the native `LoraLoader`.

### Does it affect GGUF quantization precision?

No. Patches are added on top of the dequantized float weights, the same numerical path as
native LoRA on a safetensors model. The measured per-layer error is about `4.8e-3`, purely
from bf16 rounding.

### Does inference get slower?

No. The split happens only at load time, and the slicing is a zero-copy view; nothing extra
runs during inference. The LoRA patch cost itself is the same as native loading.

---

## Limitations

- Only works inside ComfyUI; exported workflows will depend on this node. If you need it in
  other software, use `lora_repack.py split` to repack offline.
- Doesn't handle "fused model + unfused LoRA". ComfyUI core already handles that direction,
  and splitting again would break it.
- Never modifies or writes any LoRA file.

---

## Code layout

If you just want to use it, the above is enough. If you want to change the code, here's how the
files are split:

| File | Responsibility |
|---|---|
| `nodes.py` | the two node definitions, inputs/outputs, and ComfyUI registration |
| `gate_up.py` | key remapping, and the `gate_up` to `gate_layer` / `proj` split |
| `lora_keys.py` | LoRA key suffix/prefix parsing and lookup |
| `report.py` | report text formatting |
