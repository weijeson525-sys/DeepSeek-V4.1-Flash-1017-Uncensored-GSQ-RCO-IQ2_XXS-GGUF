# DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-IQ2_XXS-GGUF — 公开资料镜像

> 这是 Hugging Face 仓库的公开资料镜像（模型卡 + 小文件 + 下载脚本），**不含权重文件**。

- 来源 HF 仓库：https://huggingface.co/Abiray/DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-IQ2_XXS-GGUF
- 榜单来源：AI探路者Tim X 帖 https://x.com/aiexplorertim/status/2118947309992260083（“全球下载量最高的 10 个无审查 AI 模型”）
- 核实日期：2026-11-09；模型卡抓取日期：2026-11-10
- 许可证：MIT（许可证备注：MIT，HF cardData 核实；基底 ISTA-DASLab GSQ-RCO 版 → deepseek-ai/DeepSeek-V4.1-Flash-1017）
- 门控状态：gated=false（HF 仓库本身不门控，可匿名下载，无须令牌）
- 权重文件不在本仓库：GitHub 单文件上限 100MB，本仓库最小量化也要几 GB（首选量化 IQ2_XXS 约 138.6GB），放不下；用本仓库的 `download.sh` 从 HF 拉取。

## 量化与下载

- 首选量化：**IQ2_XXS，约 138.6GB**（5 个分片 00001–00005，文件名为 of-00006；**注意：00006 分片在 HF 文件列表中不存在，5 个分片即为完整集合**，合计 138,581,228,864 字节）
- 备用量化：无（本仓库只有一个量化）
- 另有 `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` 110.6GB：**用途未核实，可能只是 engram 部件而非完整模型——只列在量化表里并注明用途未核实，不作默认、不推荐**。
- 另有 mmproj BF16 972MB（可选，`./download.sh --mmproj` 另行下载）
- 下载方法：`./download.sh`（默认首选量化 IQ2_XXS，一次下全 5 个分片）；`./download.sh --list` 列出全部可用量化与大小；`./download.sh --mmproj` 另下 mmproj。下载后逐个核对字节数与 sha256（LFS oid）。
- **磁盘空间提醒：首选量化约 138.6GB，下载前请确保磁盘有 140GB 以上空闲空间。**

### 全部 GGUF 文件（HF 文件树实测，仅作下载清单，本仓库不含这些文件）

| 文件 | 字节数 | 大小 | sha256（LFS oid） |
|---|---|---|---|
| `DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00006.gguf` | 5,780,704 | 0.01 GB | `f356f198561f56df36f7ab92d7238a926646d76de992c34f15bcb939a5f529b9` |
| `DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00002-of-00006.gguf` | 39,661,282,624 | 39.66 GB | `db021c117306070573a8aba333f34f9c47ee757a1072aaacc0cab7bf4c737862` |
| `DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00003-of-00006.gguf` | 39,851,072,320 | 39.85 GB | `5c6de54009714a6996985f86553348e5d8e0d5ece7f64a89558e2f53d90bfa76` |
| `DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00004-of-00006.gguf` | 39,111,455,584 | 39.11 GB | `706de34f9c252234cd349081b3b76bf69dc77677ac8d006f7d4538558ba49e87` |
| `DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00005-of-00006.gguf` | 19,951,637,632 | 19.95 GB | `85e29c00f0d5616de05d20b96e367ed9eb3ba20ed958c860da899dad8c9f7a8e` |
| `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` | 110,595,290,816 | 110.60 GB | `3dac4c54893ad7aa89103c0028500eadc78e74112cef3a2930a97e67dc9fb4f3` |（用途未核实，可能只是 engram 部件而非完整模型，不作默认、不推荐）

### mmproj 文件（可选）

| 文件 | 字节数 | 大小 | sha256（LFS oid） |
|---|---|---|---|
| `mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf` | 972,208,096 | 0.97 GB | `0fdc4b7693f287e9665940cf42bd39c546ffe4f0def89c569d7d0e45d60d2df1` |

---

下面是 HF 原仓库模型卡原文（开头的 YAML frontmatter 已去掉，其余逐字照录）。

---


# DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-IQ2_XXS-GGUF (Zero-LoRA, llama.cpp b11870+ & Strata 6-Shard Layout)

This repository provides a **standalone, weight-baked abliterated (uncensored) GGUF build** of [ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF) (`IQ2_XXS`).

> **Update (October 2026):** Upgraded to the official **`llama.cpp` (`b11870+` / `v0.8.0+`) & `Strata` 6-shard `deepseek41` layout** (`00001-of-00006.gguf` through `00006-of-00006.gguf`) with embedded Jinja chat template, `deepseek41.attention.kv_source_layers` metadata, and `mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf` vision support.

## Key Highlights
* **Zero LoRA Required at Runtime:** All 86 abliterated residual-writing tensors are baked directly into the quantized GGUF weights (`blk.0..39.attn_output_b.weight` upgraded to `Q6_K`, `blk.0..39.ffn_down_shexp.weight`, and 6 `ffn_down_exps.weight` layers).
* **Native `llama.cpp` (`b11870+` / `b12017+`) & `Strata` Support:** Loads directly in mainline `llama.cpp` (`llama-cli` / `llama-server`) and `strata-dsv41` without external adapters.
* **Same Size as Original (138.58 GB vs. 138.24 GB):** Adds only `+0.34 GB` (`+0.25%`) across the entire 5-shard backbone.
* **100% GSQ-RCO Intelligence Preserved:** All 28 1-bit `IQ1_M` GSQ-calibrated routed-expert layers and the 110.6 GB `IQ4_NL` Engram lookup table remain 100% untouched byte-for-byte.

## Empirical Verification & `llama-perplexity` KL-Divergence Audit (`b12017`)

Verified end-to-end against stock `ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF` (`IQ2_XXS`) across all 40 layers and all 129,280 vocabulary logits using `llama-perplexity --kl-divergence` (`build b12017`):

| Metric | Result | Notes |
| :--- | :--- | :--- |
| **Total Model Tensors Verified** | `1,043 / 1,043` | 1,042 backbone tensors (Shards 2–5) + 1 concatenated Engram tensor (Shard 6) |
| **Abliterated Residual Writers (`blake2b`)** | `86` (80 dense + 6 exact 2-bit expert layers) | Zero 1-bit `IQ1_M` GSQ routed-expert tensors touched |
| **Preserved 1-Bit `IQ1_M` GSQ Expert Layers** | `28 / 28 (100%)` | Preserves ISTA-DASLab's 100% AIME25 / GPQA-D calibration |
| **Log-Perplexity Correlation `Cor(ln(PPL(Q)), ln(PPL(base)))`** | **`99.84%`** | `PPL(Q) = 2.4857 ± 0.5017` vs. `PPL(base) = 2.3538 ± 0.4522` |
| **129,280-Vocab Top-1 Token Agreement (`Same top p`)** | **`98.425% ± 1.109%`** | Exact greedy Top-1 match across full 40-layer forward pass |
| **End-to-End Median KL-Divergence (`Median KLD`)** | **`0.000421 nats`** | `Mean KLD = 0.009546 ± 0.001921 nats` (`1% KLD = 0.000002 nats`) |
| **Median Token Probability Shift (`Median Δp`)** | **`-0.003%`** | `Mean Δp = -0.602 ± 0.239%`, `RMS Δp = 2.747 ± 0.500%` |
| **Mean Non-Refusal Subspace Cosine Similarity** | **`0.999609`** | Across 5,119-D orthogonal intelligence subspace |

### Raw `llama-perplexity --kl-divergence` Output (`b12017`)
```text
chunk             PPL               ln(PPL(Q)/PPL(base))          KL Divergence              Δp RMS            Same top p
   1       2.4857 ±    0.5017       0.05455 ±    0.01477       0.00955 ±    0.00192     2.747 ±  0.500 %    98.425 ±  1.109 %

====== Perplexity statistics ======
Mean PPL(Q)                   :   2.485713 ±   0.501683
Mean PPL(base)                :   2.353761 ±   0.452226
Cor(ln(PPL(Q)), ln(PPL(base))):  99.84%
Mean ln(PPL(Q)/PPL(base))     :   0.054545 ±   0.014768
Mean PPL(Q)/PPL(base)         :   1.056060 ±   0.015596
Mean PPL(Q)-PPL(base)         :   0.131953 ±   0.056319

====== KL divergence statistics ======
Mean    KLD:   0.009546 ±   0.001921
90.0%   KLD:   0.029794
Median  KLD:   0.000421
10.0%   KLD:   0.000010
 5.0%   KLD:   0.000006
 1.0%   KLD:   0.000002
Minimum KLD:   0.000001
```

## How to Download & Run (No LoRA Needed — `llama.cpp` & `Strata`)

> **Important:** Both `llama.cpp` (`b11870+`) and `Strata` open the shared **`DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf`** (110.6 GB) lookup file as **Shard 6 (`...-00006-of-00006.gguf`)**. After downloading, create a symlink (Linux/macOS) or hardlink (Windows) so Shard 6 is linked automatically:

```bash
# 1. Download all 5 backbone shards + the Engram lookup file + mmproj
hf download Abiray/DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-IQ2_XXS-GGUF \
  --local-dir ./DeepSeek-V4.1-Uncensored

cd ./DeepSeek-V4.1-Uncensored

# 2A. Link the Engram file as Shard 6 (Linux / macOS):
ln -s DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00006-of-00006.gguf

# 2B. Or on Windows (Command Prompt / PowerShell):
cmd /c mklink /H DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00006-of-00006.gguf DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf
```

### Option 1: Run with `llama.cpp` (`b11870` / `v0.8.0` or newer) — Recommended
```bash
# CLI Inference
llama-cli -m DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00006.gguf \
  -ngl 99 --reasoning on --reasoning-effort 50 \
  -p "Explain mixed-precision quantization."

# Server Inference (with Vision Projector)
llama-server -m DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00006.gguf \
  --mmproj mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf \
  --jinja -ngl 99 --reasoning on --reasoning-effort 50 --port 8080
```

### Option 2: Run with Strata (1x 12 GB+ NVIDIA GPU)
> **Note for Strata `0.2.8` (`build 0.1.54`):** Always pass `--family deepseek41 --model ist-das-2.00bpw` and the **directory path** (`--model-path ./DeepSeek-V4.1-Uncensored`). Because this uncensored build upgrades `blk.0..39.attn_output_b.weight` to `Q6_K` (`138.58 GB` vs. stock `138.24 GB`), if `Strata 0.1.54`'s hardcoded stock file-size check in `setup.py` reports `"this is not a model size Strata knows"`, either connect Strata to `llama-server` above (`http://127.0.0.1:8080`) as a **Custom Model API**, or build the pack directly via Strata's bundled `tools/dsv41/pack_gguf.py`.

```bash
curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh | sh
strata launch --family deepseek41 --model ist-das-2.00bpw --model-path ./DeepSeek-V4.1-Uncensored --full-quality
```

## Upgrading Existing Pre-Oct 27 4-Shard Downloads Locally (No Re-Download Needed)
If you previously downloaded the `00001-of-00004.gguf` layout, you can convert your local folder into the 6-shard `00001-of-00006.gguf` layout in place using the included patched `convert_ds41_upstream.py` script:
```bash
python convert_ds41_upstream.py /path/to/your/model/folder --delete-old
```

## Shard Integrity & SHA256 Checksums

| Filename | Size | SHA256 |
| :--- | :--- | :--- |
| `DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00006.gguf` | 5.78 MB (5.51 MiB) | `f356f198561f56df36f7ab92d7238a926646d76de992c34f15bcb939a5f529b9` |
| `DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00002-of-00006.gguf` | 39.66 GB (36.94 GiB) | `db021c117306070573a8aba333f34f9c47ee757a1072aaacc0cab7bf4c737862` |
| `DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00003-of-00006.gguf` | 39.85 GB (37.11 GiB) | `5c6de54009714a6996985f86553348e5d8e0d5ece7f64a89558e2f53d90bfa76` |
| `DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00004-of-00006.gguf` | 39.11 GB (36.43 GiB) | `706de34f9c252234cd349081b3b76bf69dc77677ac8d006f7d4538558ba49e87` |
| `DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00005-of-00006.gguf` | 19.95 GB (18.58 GiB) | `85e29c00f0d5616de05d20b96e367ed9eb3ba20ed958c860da899dad8c9f7a8e` |
| `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` | 110.60 GB (103.00 GiB) | `3dac4c54893ad7aa89103c0028500eadc78e74112cef3a2930a97e67dc9fb4f3` |
| `mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf` | 0.97 GB (0.91 GiB) | `0fdc4b7693f287e9665940cf42bd39c546ffe4f0def89c569d7d0e45d60d2df1` |

## Acknowledgements & Credits
This release builds directly upon research, quantization breakthroughs, and tooling from across the open-source AI ecosystem:
* **DeepSeek AI** — For open-sourcing the base DeepSeek-V4.1 architecture and foundation weights.
* **ISTA-DASLab** — For developing the GSQ-RCO quantization pipeline, preserving the 1-bit `IQ1_M` routed experts with zero degradation on AIME25/GPQA-D, and providing the shared `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` lookup backbone.
* **AtomicChat** — For identifying and isolating the orthogonal refusal directions (EN+RU) that enabled weight-baked abliteration without requiring runtime LoRA adapters.
