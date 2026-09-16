# Coding guidelines (local fork)

Read before writing or building code. Agent rules: AGENTS.md.

## Build and test

- build/: Release, CUDA, BUILD_SHARED_LIBS=ON. Always CMAKE_CUDA_ARCHITECTURES=75;89 (RTX 2070 SUPER + RTX 4070).
- Rebuild only what changed: `cmake --build build --target llama-server`. After a change to common/*.h also run the full `cmake --build build` (other tools would load a mismatched libllama-common.so; ~20 s, no CUDA recompiles).
- Before a rebuild replaces a working build/bin: copy it to a backup folder (pre-<feature>-<commit>/, with a README.md).
- No builds or GPU work while a benchmark runs.
- First checks on CPU with a small model (`-ngl 0`, on a port no running server uses), so the GPUs stay free. Default: Qwen3.5-4B (3.5 GB; same tokenizer, chat template and hybrid SSM / attention layout as Flash-Next, but dense, so MoE and expert-cache paths need a real model).
- Then validate on the real model (load-log buffer math, fresh prompt and continuation, read the output). Speed claims need a session-level A/B comparison; a single llama-bench or llama-perplexity number is not evidence on this machine.
- ggml changes: run `test-backend-ops` (CPU vs CUDA); a new or changed operator gets test cases there.
- Format only the lines you changed: `git clang-format` (repo .clang-format). clangd is available for navigation.

## C++ style

- No new dependencies. New files or headers only with the user's approval.
- Simple C++: basic `for` loops, no fancy modern STL constructs, no templates.
- 4 spaces, brackets on the same line, `void * ptr`, `int & a`, no trailing whitespace. Vertical alignment where it helps reading and batch edits.
- Sized integer types (`int32_t`) in the public API; `size_t` for allocation sizes and byte offsets.
- `struct foo {}`, not `typedef struct foo {} foo`. In C++ omit optional `struct` / `enum` keywords: `llama_context * ctx;`, `const llama_rope_type rope_type;`.
- Follow the surrounding code; for anything not covered, the C++ Core Guidelines.
- Close preprocessor blocks with a comment: `#endif // FOO`.
- New code follows these rules (legacy code may not yet). Exceptions are allowed in isolated backend-specific code that does not interface with ggml.
- If you had to read the source to learn how to use an API, add a short summary to its header. Fix incorrect or outdated docs you notice.

## Naming

- `snake_case` for functions, variables and types. Longest common prefix first: `number_small`, `number_big`, not `small_number`.
- Enum values upper case, prefixed with the enum name: `LLAMA_VOCAB_TYPE_BPE`.
- Functions are `<class>_<method>`, with `<method>` = `<action>_<noun>`: `llama_model_init`, `llama_sampler_chain_remove`, `llama_sampler_get_seed`, `llama_adapter_lora_free`. `get` and the noun can be omitted; the `_context` suffix of the class is optional, used to disambiguate (`llama_set_embeddings`, `llama_n_threads` belong to `llama_context`); `init` / `free` for constructor / destructor.
- `_t` suffix for types opaque to the user: `typedef struct llama_context * llama_context_t;`.
- C/C++ files lowercase with dashes (`.h`, `.c`, `.cpp`); Python files lowercase with underscores.

## ggml

- Tensors are row-major: dimension 0 = columns, 1 = rows, 2 = matrices.
- `C = ggml_mul_mat(ctx, A, B)` means C^T = A B^T, i.e. C = B A^T (media/matmul.png).
- Intro examples: https://github.com/ggml-org/ggml/tree/master/examples (simple, gpt-2, mnist).

## Server

- Read tools/server/README-dev.md (architecture) before server changes.
