#!/usr/bin/env bash
# One phase = one branch + one commit + one push + one tag.
#
#   make git-phase PHASE=1 SLUG=log-corpus MSG="feat(corpus): add labelled synthetic generator"
#   bash scripts/git_phase.sh 1 log-corpus "feat(corpus): ..."
#
# After it finishes, open the PR on GitHub, review the diff, squash-merge into main,
# then push the tag:  git push origin --tags
set -euo pipefail

PHASE="${1:?usage: git_phase.sh <phase> <slug> <commit message>}"
SLUG="${2:?usage: git_phase.sh <phase> <slug> <commit message>}"
MSG="${3:?usage: git_phase.sh <phase> <slug> <commit message>}"

BRANCH="phase/${PHASE}-${SLUG}"
TAG="v0.${PHASE}.0-phase${PHASE}"

if ! git rev-parse --git-dir >/dev/null 2>&1; then
  echo "[git] not a repository yet — run: git init -b main && git remote add origin <url>"
  exit 1
fi

echo "[git] branch: $BRANCH"
git checkout -b "$BRANCH" 2>/dev/null || git checkout "$BRANCH"

echo "[git] staging"
git add -A

if git diff --cached --quiet; then
  echo "[git] nothing staged — did you forget to save your files?"
  exit 1
fi

git status --short

echo "[git] commit: $MSG"
git commit -m "$MSG"

echo "[git] push"
git push -u origin "$BRANCH"

if git tag -l "$TAG" | grep -q .; then
  echo "[git] tag $TAG already exists — skipping"
else
  git tag -a "$TAG" -m "Phase ${PHASE}: ${MSG}"
  echo "[git] tagged $TAG (push with: git push origin $TAG)"
fi

cat <<EOF

──────────────────────────────────────────────────────────────
NEXT (do this on GitHub, not here):
  1. Open the PR:  $BRANCH -> main
     gh pr create --fill --base main --head $BRANCH   (or use the web UI)
  2. Confirm CI is green (Actions tab).
  3. Squash-merge into main.
  4. Locally:
       git checkout main && git pull
       git push origin $TAG
  5. Update docs/HANDOFF.md with the phase result and commit that on main.
──────────────────────────────────────────────────────────────
EOF
