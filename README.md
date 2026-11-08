# ComfyUI-UnfusedQwenImageLoraFix

给 Qwen-Image 2.1 用的 LoRA 加载节点。功能上和 ComfyUI 自带的 `LoraLoader` 一样，
MODEL 和 CLIP 两条路都能接。区别是它会顺手修一个键名对不上的问题：不修的话，
一部分 LoRA 权重会被悄悄丢掉，而且不会报错。

节点不改动磁盘上的 LoRA 文件，也不引入额外依赖，用的全是 ComfyUI 自带的加载和
打补丁流程。

---

## 问题出在哪

Qwen-Image 2.1 的 SwiGLU 层有两种等价的权重存法：

| 布局 | 权重名 | 形状 |
|---|---|---|
| fused（合着存） | `img_mlp.gate_up` | `[24576, 4096]`，前一半是 gate，后一半是 up |
| unfused（拆开存） | `img_mlp.gate_layer` + `img_mlp.proj` | 各 `[12288, 4096]` |

这两种写法在数学上是同一个东西，只是矩阵怎么摆的问题。麻烦在于生态里两边都有：
GGUF 量化版基本都是拆开存的，而 kohya / sd-scripts 训出来的 Qwen 2.1 LoRA 基本
都是合着存的。

ComfyUI 核心只处理了一个方向：模型合着存、LoRA 拆开存时，它会自动按行切片。反过来，
模型拆开存、LoRA 合着存时，LoRA 里的 `gate_up` 在模型里找不到对应的权重，整块就被
跳过了。

| 模型 | LoRA | 原生加载的结果 |
|---|---|---|
| fused | unfused | 自动切片，正常 |
| fused | fused | 正常 |
| unfused | unfused | 正常 |
| **unfused** | **fused** | **整块 `gate_up` 被丢弃** |

最后这一行的影响不算小。Qwen 2.1 LoRA 通常有 224 个模块，其中 32 个是
`img_mlp.gate_up`，大约占整个 LoRA 参数量的 37%。控制台只会留下一行
`lora key not loaded`，画面上不会报错，表现是"LoRA 确实起作用了，但没那么像"——
这种问题最难查，因为一切看起来都正常。

## 节点怎么修

加载时把 `gate_up` 的 `lora_B` 按输出行切成两半，`lora_A` 两边共用：

```
delta_gate = B[:H] @ A        # B: [2H, r]
delta_up   = B[H:] @ A
delta_gate + delta_up == B @ A
```

两个结果拼起来和原来的 `B @ A` 完全相等，所以 rank 不变，也没有近似误差。切完的
键再交给 ComfyUI 原生的 `load_lora`，后续打补丁的流程和自带节点走的是同一条路。
LoRA 文件本身不会被修改或重写。

---

## 安装

把整个目录放进 ComfyUI 的 `custom_nodes`：

```bash
cp -r ComfyUI-UnfusedQwenImageLoraFix ~/ComfyUI/custom_nodes/
```

如果想让仓库更新直接生效，可以用软链接：

```bash
ln -s "$(pwd)/ComfyUI-UnfusedQwenImageLoraFix" ~/ComfyUI/custom_nodes/ComfyUI-UnfusedQwenImageLoraFix
```

重启 ComfyUI 后，节点会出现在 **Qwen-Image 2.1** 分类下。不需要装额外的 Python 包。

---

## 节点一：Qwen-Image 2.1 LoRA Loader (auto gate_up fix)

内部 id 是 `Qwen21LoraAutoFix`。这是平时用的主节点。

### 输入

| 名称 | 类型 | 说明 |
|---|---|---|
| `lora_name` | 下拉框 | `models/loras` 里的文件 |
| `strength_model` | FLOAT | 扩散模型强度，默认 1.0 |
| `strength_clip` | FLOAT | 文本编码器强度，默认 1.0 |
| `model` | MODEL（可选） | 扩散模型，GGUF 也接这里 |
| `clip` | CLIP（可选） | 文本编码器，接 `CLIPLoader` 的输出 |
| `strict` | BOOLEAN | 打开后，只要有一个模块映射不上就直接报错，默认关闭 |

`model` 和 `clip` 至少接一个。只接 `model` 时，用法和 `LoraLoaderModelOnly` 一样；
两个都接时，用法和原生 `LoraLoader` 一样，行为也一致。

### 输出

| 名称 | 类型 | 说明 |
|---|---|---|
| `model` | MODEL | 打好补丁的模型；没接就原样返回 `None` |
| `clip` | CLIP | 打好补丁的 CLIP；没接就是 `None` |
| `report` | STRING | 映射报告，建议接一个 `Show Text` 看一眼 |

### 报告怎么看

加载一个只训练了扩散模型的 LoRA，输出大概是这样：

```
[Qwen21LoraAutoFix] zzz_chibi_v2_st2500.safetensors
  模型 MLP 布局 : gate_layer + proj (拆分)
  LoRA 模块     : 192
    扩散模型侧  : 192
    文本编码器侧: 0
  无损拆分      : 32 个 gate_up -> gate_layer + proj
                  例: diffusion_model.transformer_blocks.0.img_mlp.gate_up
  model patch   : 192
  clip patch    : 0
  patch 条目    : 224
```

各字段的意思：

| 字段 | 含义 |
|---|---|
| 模型 MLP 布局 | `gate_up` 表示模型是合着存的；`gate_layer + proj` 表示拆开存的 |
| 扩散模型侧 / 文本编码器侧 | 这个 LoRA 里分别有多少模块属于两边 |
| 无损拆分 | 这次实际拆开了多少个 `gate_up`；是 0 就说明不需要拆 |
| 未映射 | 确实对不上的模块，会逐条列出来 |
| model / clip patch | 分别有多少个 patch 落到了模型上 |
| patch 条目 | 去重之后的 patch 总数 |

正常情况下 **未映射应该是 0**。如果 `model patch` 比「扩散模型侧」少很多，说明模型里
没有对应的权重，先检查是不是拿错了模型版本——Qwen-Image 1.0 是 60 层，2.1 是 32 层，
两者的键名并不通用。

---

## 节点二：Qwen-Image 2.1 LoRA Audit

内部 id 是 `Qwen21LoraAudit`。它只做体检，不修改模型，报告会同时显示在节点上和
控制台里。跑图之前可以先拿它确认三件事：

- 这个 LoRA 是 fused 还是 unfused；
- 当前 GGUF 模型是哪种布局；
- 如果直接用原生加载，大概会丢多少。

用法很简单：把 `MODEL` / `CLIP` 接进来，选好 `lora_name`，然后把 `report` 接到
`Show Text`。

---

## 典型接法

**只接扩散模型**（最常见，绝大多数 Qwen-Image LoRA 都属于这种）

```
Unet Loader (GGUF) ──┐
                     ├─ Qwen21LoraAutoFix ─ model ─ KSampler
CLIPLoader ──────────┘   (clip 不接)
```

**MODEL + CLIP 双路**（LoRA 里带文本编码器权重时）

```
Unet Loader (GGUF) ── model ─┐
                             ├─ Qwen21LoraAutoFix ─ model ─ KSampler
CLIPLoader ────────── clip ──┘        └──── clip ── CLIP Text Encode
```

**多个 LoRA 叠加**

直接把多个 `Qwen21LoraAutoFix` 串起来就行，每个节点用各自的 `strength_model`。
这和串联原生 `LoraLoader` 没有区别，patch 会正常累加。

---

## 常见问题

### 能直接替代原生 LoraLoader 吗？

可以。这个节点内部走的就是 `comfy.lora.model_lora_keys_unet`、
`model_lora_keys_clip`、`load_lora` 和 `add_patches` 这一套，只是中间多了一步
键名规范化。如果你的模型和 LoRA 本来就完全匹配，继续用原生节点也完全没问题。

### 为什么不直接把 LoRA 文件重打包？

也可以，那是 `lora_repack.py split` 的做法：离线改文件，任何前端都能用。这个节点
的好处是不动原文件，换 LoRA、换强度都方便，也不会多出一份占空间的副本。

### 接了 CLIP，但报告里「文本编码器侧」是 0？

说明这个 LoRA 只训练了扩散模型。Qwen-Image 的绝大多数 LoRA 都是这样。接 CLIP 不会
影响结果，只是 `strength_clip` 没有实际作用。

### strict 什么时候该打开？

调参或者试新 LoRA 的时候。比如你想确认某个 LoRA 和当前模型是否完全匹配，打开
`strict` 后，报错信息会把所有对不上的模块名列出来。跑正式工作流时建议关掉，免得
个别模块导致整个流程失败。

### 模型本来就是 fused 布局，还需要这个节点吗？

不需要。ComfyUI 核心会自动处理 fused 模型加 unfused LoRA 的情况，再拆一次反而会错。
可以先用 `Qwen21LoraAudit` 看一眼「模型 MLP 布局」，如果显示 `gate_up (融合)`，
直接接原生 `LoraLoader` 就行。

### 会影响 GGUF 的量化精度吗？

不会。patch 是在反量化之后的 float 权重上叠加的，和原生 LoRA 在 safetensors 模型上的
数值路径一致。实测单层误差大约 `4.8e-3`，纯粹是 bf16 舍入造成的。

### 推理会变慢吗？

不会。拆分只发生在加载时，而且切片是零拷贝的 view；推理阶段不会多出任何计算。
LoRA patch 本身的成本和原生加载一样。

---

## 局限

- 只在 ComfyUI 内部生效，导出的工作流会依赖这个节点。如果要给别的软件用，请走
  `lora_repack.py split` 离线重打包。
- 不处理「fused 模型 + unfused LoRA」。这个方向 ComfyUI 核心已经处理好了，再拆一次
  反而会出错。
- 不修改、不落盘任何 LoRA 文件。

---

## 代码结构

如果只是想用，上面这些就够了。想改代码的话，文件分工是这样的：

| 文件 | 职责 |
|---|---|
| `nodes.py` | 两个节点的定义、输入输出和 ComfyUI 注册 |
| `gate_up.py` | 键重映射，以及 `gate_up` 到 `gate_layer` / `proj` 的拆分 |
| `lora_keys.py` | LoRA 键名的后缀、前缀解析和查找 |
| `report.py` | 报告文本的格式化 |
