# llama-spillway

A [llama.cpp](https://github.com/ggml-org/llama.cpp) fork for running a **76 GiB hybrid mixture-of-experts model on a desktop with 32 GB of DDR5 RAM and 20 GB of VRAM**, split over two mismatched GPUs (RTX 4070 and RTX 2070 SUPER), with the rest of the model memory-mapped from NVMe.

Non-optimal, to say the least. The model in question is Qwen3.8-Flash-Next, UD-IQ3_XXS quant (76.3 GiB).

Inspired by (and after trying out) [thecodacus's fork](https://github.com/thecodacus/llama.cpp), which lets you run huge models on modest hardware, I wanted to see how much further I could push the performance on my own hardware. What started as plain launch config tuning eventually became its own fork, **going from a prefill that swung wildly between 4 and 64 tokens/s on the same prompt, with decode at 8-10 tokens/s, to a now steady 90-110 tokens/s prefill and 22-25 tokens/s decode.**

Since the model does not fit in memory, every layer of the stack matters: which GPU holds which experts, how the kernel reads the model file, how the server reuses its prompt cache, and how the hardware power state follows the load. This fork changes the code where measurement showed a gain, and documents the launch settings that worked on my particular setup. Experimental items and items still under validation are marked as such.

This is an ongoing project. `main` is a snapshot of the working branch and is updated from time to time.

Also, this README functions both as a technical registry of code changes and a log of which configs worked best for me, in case that information proves useful to anybody with a similar setup.

## What this fork adds

### 1. Per-GPU placement of the expert cache

The base fork keeps the most-used experts of each layer in GPU memory, but always on the first GPU. With the model split over two GPUs, the layers on the second GPU had to fetch their cached experts from the other card. This fork puts each layer's cache on the GPU that runs that layer.

**Result: decode graph splits went from 135 to 99 and GPU-to-GPU copies per token from 58 to 4, with no measurable change in prefill speed.**

### 2. `LLAMA_MMAP_SEQ=auto`: read the model file sequentially, but only when it pays off

**This was the biggest find for my setup by a mile.**

When the model is bigger than RAM, parts of it are read from the NVMe drive on demand. Linux normally reads ahead in larger chunks, but after enough misses it gives up and reads one small 4 KiB page at a time, which made prefill crawl. Telling the kernel to read sequentially fixes prefill, but applied all the time it made decode almost 9 times slower, because a single generated token only needs a small part of the model.

With `LLAMA_MMAP_SEQ=auto`, the fork turns the sequential hint on for large prompt batches only, and leaves it off for decode. Compared with no hint, **prefill ran 4.92x faster and a whole session took 59% less time, with no measurable change in decode speed.**

**IMPORTANT: this applies only to models larger than RAM. A model that fits in the page cache is never read from disk after loading, so leave the variable unset for it.**

### 3. `--cache-text-match`: keep the prompt cache when only the tokens differ

Sometimes the model generates a word as two tokens (`" Item" + "Stack"`), but when the chat client sends the conversation back, the same text becomes one token (`" ItemStack"`). The server then sees a prompt that differs from its cache, and with this kind of model it has to go back to an older checkpoint and reprocess the whole previous turn. This happened about once every 14,000 generated tokens.

With `--cache-text-match`, the server compares the text instead of the tokens, and keeps its cache when the text matches. It does so only when the result is exactly the prompt the client sent; otherwise it falls back to the normal path. After such a split, the tokens to reprocess went from 4,033 to 31 in a web chat replay, and from 3,107 to 1,014 in an agent replay.

Output quality stayed within normal run-to-run variation across three rounds of checks (70 in total).

### 4. Server power switch

Running the CPU and GPUs in maximum-performance mode made decode 1.47-1.57x faster, but it costs about 59 W more while idle, which adds up when the server sits idle between requests during long sessions.

The power switch (`--power-switch-gpu`, `--power-switch-cpu`) turns those settings on only while the server is working, and releases them after 10 idle seconds. Decode speed matches holding the settings on by hand.

**IMPORTANT: Linux and NVIDIA only.**

### 5. Upstream changes, picked early

Several upstream fixes and speed-ups were pulled in before they reached the base fork, keeping their authorship, and checked on all three models I use:

- **Fixes:**
  - Flash-Next's attention indexer no longer allocates a cache it never uses, which frees some VRAM (#28330).
  - A CUDA sorting bug that could return garbage indices (#28389).
  - Race conditions in two CUDA matrix-multiply kernels, one of them the kernel for the experts (#28475).
  - A corner case when copying selected experts to the GPU and none are selected (#28739).
  - A synchronization bug in the f16 flash-attention kernel (#27870).
  - Some model inputs no longer force an extra split of the compute graph (#28387).
- **Numerics:** the linear-attention layers (gated delta net) now normalize their inputs exactly as the reference implementation does. This changes the output slightly (#28068).
- **Speed** (still under validation, see [Current known limitations](#current-known-limitations)):
  - The step that picks each token's top experts (the MoE router) now always runs as one fused GPU kernel. Before, it only did when the memory layout happened to allow it (#28432).
  - Flash-Next's hyper-connection steps, which mix the model's parallel residual streams around each layer, run as dedicated fused GPU kernels instead of a chain of small operations (#28901).
  - The normalization and scaling inside those hyper-connection steps are written so the GPU can merge them into one kernel (#28896).
  - Together, a first check on Flash-Next showed 1.10x faster short decode.
- **Experimental:** with `--backend-sampling`, the server rebuilds its GPU scheduler for every request, freeing and reallocating all compute buffers (0.6-0.9 s per request on Flash-Next). `LLAMA_SCHED_KEEP=1` keeps the scheduler when nothing has changed. A port of the still-open upstream #28872.

Each of the three speed-ups that change the output has a switch to turn it off (see [Environment variables](#environment-variables)).

### 6. Tests and docs

New backend tests at the real model shapes, and regenerated flag tables in the server, CLI and completion READMEs.

## Launch tuning (no code change)

- **Larger prompt batches (`-ub 1024`)** mean fewer passes over the model on disk: prefill ran 1.70x faster and a session took 22% less time, with no measurable change in decode. Output quality passed all checks (28 total).
- **128K context** (up from 64K): doubling the context itself costs nothing measurable. The cost comes from making room for it in VRAM. The 128K launch moves two more expert layers to the CPU (44 instead of 42) and shifts more of the model onto the RTX 4070 (`-ts 33.5,15.5` instead of `3,2`). It also runs CPU work alongside the GPUs (`--sched-async-cpu`) and keeps prompts under 1024 tokens from copying experts to the GPU (`GGML_OP_OFFLOAD_MIN_BATCH=1024`). Against the 64K launch, decode is 12-13% slower and prefill of a long prompt 9% slower, while a short first prompt finishes in about half the time.
- **No speculative decoding with the model's built-in draft head (MTP)** as it makes Flash-Next slower on my setup (decode 8-29% slower in long sessions): the head sits in CPU memory and adds 20-66% more reads per token, and most experts already come from NVMe. On Qwen3.6-35B-A3B, which fits in RAM, it makes decode 1.28-1.43x faster on code review text.
- **Mixed K / V cache types** (for example `q8_0` for K and `q4_0` for V) need the build option `GGML_CUDA_FA_ALL_QUANTS=ON`. Without it, attention silently runs on the CPU.
- **Other models:** on Qwen3.8 27B, `-fit` leaves the first layer on the CPU, because it counts one layer fewer than the loader does. Placing every layer on the GPUs by hand (`-fit off -ngl 99 -ts 41.5,24.5`) made prefill 1.76x faster (636 to about 1,130 tokens/s) and decode about 4-6% faster.

## How it was measured

Every number here comes from A/B comparisons on just my machine:

- Each session is a fresh server launch, with the file cache cleared first, running the same fixed requests.
- The two setups alternate (for example A B B A B A A B), so drift over time cancels out.
- Sessions are compared statistically, with confidence intervals. Adopted changes use 4 or 8 sessions per side; quick checks use 2 and only show a direction.
- Changes that alter the model's output must first pass a quality check against normal run-to-run variation.
- Single `llama-bench` runs are not used: on my desktop, disk read behaviour varies too much between runs.

Details: [Technical details](#technical-details).

## Hardware and software

- CPU: AMD Ryzen 7 9800X3D, 32 GB DDR5-6000 RAM.
- GPUs: NVIDIA RTX 4070 12 GB (compute 8.9, PCIe Gen4 x8) and RTX 2070 SUPER 8 GB (compute 7.5, PCIe Gen3 x4).
- Storage: WD SN850X 2 TB NVMe (model files).
- OS: CachyOS, Linux 7.2.6, NVIDIA driver 615.71.09, CUDA 13.4.1, g++ 15.
- Models: Qwen3.8-Flash-Next UD-IQ3_XXS (main target), Qwen3.8 27B and Qwen3.6-35B-A3B (both hybrid SSM / attention; the 27B has a dense FFN, the 35B is MoE).

## Build

```sh
cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_CUDA_ARCHITECTURES="75;89" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Set `CMAKE_CUDA_ARCHITECTURES` to your GPUs. See [docs/build.md](../docs/build.md) for other backends.

## Example launch (Flash-Next, 64K context)

```sh
LLAMA_MMAP_SEQ=auto ./build/bin/llama-server \
  -m /path/to/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf \
  --moe-cache-profile /path/to/routing-profile.csv --moe-cache-slots 48 \
  -ngl 99 --n-cpu-moe 42 \
  -ot "blk\.4[2-4]\.ffn_.*_exps\.weight=CUDA0,blk\.4[5-7]\.ffn_.*_exps\.weight=CUDA1" \
  --no-sched-async-cpu -t 8 --threads-batch 12 \
  -sm layer -ts 3,2 \
  --load-mode mmap -fit off -fa on -ctk q8_0 -ctv q8_0 \
  -c 65536 -np 1 -b 2048 -ub 1024 --jinja \
  --power-switch-gpu --power-switch-cpu \
  --cache-text-match
```

The placement (`--n-cpu-moe`, `-ot`, `-ts`, cache slots) is fitted to my desktop's VRAM; start from your own load log. The routing profile comes from `llama-moe-trace` ([thecodacus's fork](https://github.com/thecodacus/llama.cpp)). Please refer to it for more information on `llama-moe-trace` and the `--moe-cache-profile` and `--moe-cache-slots` flags.

## Flags and environment variables

### Server flags

| Flag | Default | Effect |
|---|---|---|
| `--cache-text-match` | off | keep the cached tokens where the prompt text matches the cache (single slot, text-only prompts) |
| `--power-switch-gpu` | off | hold NVIDIA's "prefer maximum performance" mode on all GPUs while the server has work (Linux) |
| `--power-switch-cpu` | off | hold the power-profiles-daemon `performance` profile while the server has work (Linux) |
| `--power-switch-idle SECONDS` | 10 | idle time before the power switch releases its settings |
| `--power-switch-check SECONDS` | 0 (off) | while the server holds the CPU performance profile, check every N seconds that the hold is still in place, and log a warning if another program (for example a manual profile change) has dropped it |

Do not combine the power switch with the same settings set by hand (for example in `nvidia-settings`), or the idle release has no effect. Full generated tables: [tools/server/README.md](../tools/server/README.md).

### Environment variables

| Variable | Default | Effect |
|---|---|---|
| `LLAMA_MMAP_SEQ` | unset (off) | `auto`: large sequential reads of the model file during prompt processing only (recommended for models larger than RAM). Unset, empty, `0` or `off`: off. Any other value turns large reads on for the whole run, which makes decode almost 9 times slower on my desktop (kept for reproducing an earlier experiment). |
| `LLAMA_MMAP_SEQ_MIN_TOKENS` | derived | with `LLAMA_MMAP_SEQ=auto`: the batch size from which the hint is used (default `ceil(2 * n_expert / n_expert_used)`, 1 for dense models); `off` never uses it |
| `LLAMA_SCHED_KEEP` | 0 | `1`: keep the GPU scheduler between requests instead of rebuilding it; matters with `--backend-sampling` (port of upstream #28872). Experimental: builds and loads, not yet measured |
| `GGML_CUDA_TOPK_MOE_DEPS` | 1 | `0`: run the MoE router as one fused kernel only when the memory layout allows it, as before #28432 |
| `LLAMA_FUSED_HC` | 1 | `0`: run Flash-Next's hyper-connection steps as a chain of small operations instead of fused kernels, as before #28901 (also affects `deepseek4` models) |
| `LLAMA_HC_NORM_FUSE` | 1 | `0`: keep the normalization and scaling in Flash-Next's hyper-connection steps apart, as before #28896 |

The last three exist for A/B checks and rollback. With all three at `0`, the model runs the same computation as before those three upstream speed-ups.

## Current known limitations

- Everything was measured on one machine, [the hardware listed above](#hardware-and-software).
- `LLAMA_MMAP_SEQ=auto` and `--cache-text-match` have been validated on Flash-Next only. The 27B and 35B fit in RAM, so the read hint does not apply to them.
- At 128K, Flash-Next output quality stayed within normal variation on 11 test cases up to about 40,000 tokens, but one retrieval check near 120,000 tokens failed narrowly (0.519 against a limit of 0.5). Most of the loss there is the model leaning towards ending its turn; the cause is not yet explained. GPU memory headroom also depends on how much VRAM the desktop uses.
- 27B and 35B settings (cache types, MTP) have not been quality-checked; the MTP speedup comes from single runs and depends on the text.
- The upstream speed-ups are still being validated, and a quality re-check is pending. Their 1.10x decode figure comes from 2 sessions per side, so it only shows a direction. `LLAMA_SCHED_KEEP` has not been measured yet.
- Some questions are still open: why the kernel stops reading ahead for the model file in the first place, why it stays that way into the next request under `auto`, and what causes a CUDA crash with the base fork's `GGML_SCHED_PREFETCH_EXPERTS` in this setup (leave it unset here).
- On the 27B, any weight left on the CPU halves prefill speed. A scheduler fix for it is designed but not built.
- Samples are small (2 to 8 sessions per side), so some effects under about 25% deep into the context cannot be resolved.
- The upstream linear-attention normalization change (#28068) changes the output slightly on all three models, and has no switch.

## Technical details

**Per-GPU expert cache.** Each layer's cache pack follows the layer split (`-ts`), CPU layers use the first GPU, and there is one buffer per device. Prefill ratio 1.03 (95% CI 0.87-1.21). Code: `src/llama-model.cpp`.

**`LLAMA_MMAP_SEQ=auto`.** Prefill was fault-bound, not disk-bound: the kernel's mmap read-around turns itself off after many misses (`mmap_miss` against `MMAP_LOTSAMISS`) and then reads one 4 KiB page per fault. `MADV_SEQUENTIAL` skips that check. A 512-token batch routes to nearly every expert, so larger reads pay off; a single decode token routes to 10 experts per layer and reads 33x more bytes with the advice on (whole-run advice: 0.114x decode). `auto` arms the advice in `llama_context::decode()` for batches of at least `ceil(2 * n_expert / n_expert_used)` tokens (103 for Flash-Next) and disarms it otherwise. Measured over 4 against 4 sessions with 27,530 prompt tokens each: prefill 4.92x, session wall time 0.41x, decode within noise. Code: `src/llama-mmap.cpp`, `src/llama-context.cpp`.

**`--cache-text-match`.** At task launch, the prompt keeps the slot cache's token ids up to the point where the text still matches, only when that goes further than the token-level match, and only if the tokens after that point are the canonical ones. The assembled prompt must detokenize to the request text, or the canonical tokens are kept. Skipped for client-supplied token ids, multimodal prompts, `cache_prompt: false` and more than one slot. Code: `tools/server/server-context.cpp`.

**Power switch.** The GPU side sets PowerMizer mode 1 through NVML; the CPU side holds a power-profiles-daemon `performance` profile. Against holding the same settings by hand: decode 0.999x (95% CI 0.945-1.061). Code: `tools/server/server-power.{h,cpp}`.

**Upstream switches.** Each switch was checked to restore the earlier graph: identical op list on Flash-Next, identical greedy output on Qwen3.6-35B-A3B. `LLAMA_SCHED_KEEP` keeps the scheduler on re-reserve while `max_nodes` is unchanged, a guard the upstream change does not have.

**`-ub 1024`.** With `auto`, bytes read per ubatch stay flat while the ubatch count halves, so prefill time follows the byte count.

**Method.** Arms alternate in a balanced order (ABBA BAAB) and are compared with permutation tests and bootstrap confidence intervals; the smallest possible p at 4 against 4 is 0.029. Decode comparisons use the same context depth and generated-token count in every arm. The quality gate compares per-token KL divergence and log-likelihood on cold prompts up to 40,698 tokens against a run-to-run floor measured in the same run. VRAM is checked with load-log buffer arithmetic and a fitted VRAM model, with a fixed free-memory margin per GPU at peak. Harnesses are resumable, rehearsed on the CPU before GPU runs, and check a set of fixed conditions (power state, system settings, tool hashes) at every session. The harnesses and raw session data are not published.

## How this was built

This project was developed with [Claude Code](https://claude.com/claude-code). I set the goals, decided what to test and what to adopt, approved each source change before it was written, ran the long benchmarks, and reviewed the results. Claude, mainly Opus models, wrote most of the code, the benchmark harnesses and the analysis, under written working rules: [AGENTS.md](../AGENTS.md) and [CODING-GUIDELINES.md](../CODING-GUIDELINES.md). The model behind each of this fork's own commits is named in its `Co-Authored-By` line. The experiments are recorded in a private research log that is not part of this repository.

## Repository notes

- `main` is a snapshot of a local working branch, published without the private research log.
- Commits prefixed `local:` touch fork-only setup or policy. No commit here is proposed upstream.
- `AGENTS.md`, `CLAUDE.md` and `CODING-GUIDELINES.md` are the agent rules of the local setup, kept as they are; they refer to the local branch names.
- The root [README.md](../README.md) is the upstream llama.cpp README with the base fork's section.

## Credits and license

- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) and its contributors.
- [thecodacus/llama.cpp](https://github.com/thecodacus/llama.cpp) (`perf` branch), the base of this fork: the MoE expert cache (including its `qwen4exp` support), expert prefetch and host pinning.
- The authors of the upstream pull requests listed above; cherry-picked commits keep their authorship.
- Third-party libraries: see Acknowledgements in the root [README.md](../README.md).

MIT license, as upstream: [LICENSE](../LICENSE).
