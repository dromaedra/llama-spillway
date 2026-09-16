# Agent rules (local fork, branch perf)

Local-only fork: nothing is pushed or proposed upstream. Local commits on `perf` are fine.

## Never

- Write to any remote: `git push`, `gh pr create`, `gh pr comment`, `gh issue create`, reviews. Read-only look-ups (`gh search issues`, `gh search prs`, `gh issue view`) are fine; search before new work.
- Add a new file under `tests/` without the user's approval. Reuse existing test infrastructure; no tests for trivial features.

## Code

- Keep it simple: a change that does 90% with less complexity beats one that does 100%. Reuse existing infrastructure; avoid new subsystems and invasive changes (small diffs also rebase cleanly onto origin/perf).
- Read the relevant files first and match the surrounding patterns. Large change or new pattern: pause and ask the user.
- Build, test and style rules: CODING-GUIDELINES.md (read before writing or building code).
- ASCII only in code, comments and commit messages: `-` not an em dash, `->` not an arrow, `x` not a times sign, `...` not an ellipsis character.
- Do not wrap lines at a fixed width or split a sentence across lines.
- llama.cpp has no Minja; its Jinja engine is `common/jinja` (unnamed).

## Comments (re-read after a context compaction)

- Write the code first, then comment only where needed: 1-2 lines, ASD-STE100 simple English.
- No comment that restates the code: `n_ctx = read_metadata("context_length", 1024);` needs none.
- Comment non-obvious invariants: `task_queue->on_idle(); // also signal child disconnection`.
- Write for any future reader: `// reset here, as we will release the slot below`, never a reference to the current task or bug report.
- No new comments on code copied from elsewhere.
- Multi-line: short list lines, no fixed-width wrapping: `// at step_idx g:` / `// - read code from out_code_cache[g], embed it with codebook table g-1`.

## Where to look (load when needed)

- Build: docs/build.md. Server: tools/server/README.md, tools/server/README-dev.md. New model: docs/development/HOWTO-add-model.md.
- Parsing: docs/development/parsing.md (PEG), docs/autoparser.md, common/jinja/README.md.
- Skills (not registered; read the SKILL.md by hand): skills/add-new-model, skills/code-review (base is `origin/perf`; there is no `master`).
