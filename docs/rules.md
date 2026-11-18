# Rules

Each line: what broke -> rule.

- A delegate created `.claude/gates-off` to pass a gate -> briefs forbid `*_ALLOW` and `.claude/gates-off`; a blocked agent stops and reports.
- `perl -pi` over tracked files replaced `agent/body_engine/shiu_worker.mojo` (a symlink) with its target -> skip mode 120000 entries before in-place rewrites.
