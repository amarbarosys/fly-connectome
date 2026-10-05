#!/usr/bin/env bash
# Merge gate for lane commits: run check.sh stages on a CLEAN worktree at COMMIT (never a lane's working tree, which
# may hold uncommitted edits), data links as in the main checkout, every stage under gpu-wait.
# usage: [MERGE=1] agent/body_engine/verify_commit.sh COMMIT STAGE [STAGE ...]   -> .work/verify/<commit>-<stage>.log
# First checks that COMMIT merges into main without conflicts (no GPU time spent on a branch that must rebase first).
# MERGE=1: on PASS, merge --no-ff into main (no stash: git refuses only if uncommitted edits touch the same files).
set -euo pipefail
R=$(cd "$(dirname "$0")/../.." && pwd); C=$1; shift
if ! git -C "$R" merge-tree --write-tree main "$C" >/dev/null; then
  echo "FAIL merge: $C conflicts with main $(git -C "$R" rev-parse --short main):"
  git -C "$R" merge-tree --write-tree --name-only main "$C" | sed -n '2,/^$/p' | sed '/^$/d;s/^/  /'; exit 1
fi
W=$R/.work/verify-$C; L=$R/.work/verify; mkdir -p "$L"
git -C "$R" worktree add -q --detach "$W" "$C"
trap 'git -C "$R" worktree remove --force "$W"' EXIT
ln -s "$R/.venv" "$W/.venv"
mkdir -p "$W/.work/agent/body_engine"
ln -s "$R/.work/agent/gpu-body-port" "$W/.work/agent/gpu-body-port"
for d in spider-body mujoco-src; do [ -e "$R/.work/agent/body_engine/$d" ] && ln -s "$R/.work/agent/body_engine/$d" "$W/.work/agent/body_engine/$d"; done
fail=0; bad=""
for s in "$@"; do
  if gpu-wait run --priority 70 --shared --vram 1 --timeout 3600 -- "$W/agent/body_engine/check.sh" "$s" > "$L/$C-$s.log" 2>&1; then
    echo "PASS $s ($(grep -cE '^PASS' "$L/$C-$s.log") PASS lines)"
  else
    echo "FAIL $s: $L/$C-$s.log"; fail=$((fail+1)); bad="$bad $s"
  fi
done
[ $fail = 0 ] || { echo "FAIL $fail/$# stages:$bad"; exit 1; }
echo "PASS $C: $*"
if [ "${MERGE:-0}" = 1 ]; then
  b=$(git -C "$R" for-each-ref --contains "$C" --format='%(refname:short)' refs/heads/team | head -1)
  git -C "$R" merge --no-ff -q -m "Merge ${b:-$C} ($C): verified $*" "$C" && echo "MERGED $C -> $(git -C "$R" rev-parse --short main)"
fi
