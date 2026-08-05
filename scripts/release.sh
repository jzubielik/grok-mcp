#!/usr/bin/env bash
# Usage: scripts/release.sh 0.2.0  (called by: make release VERSION=0.2.0)
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <version>   e.g.  $0 0.2.0" >&2; exit 1
fi

NEW="$1"
TAG="v${NEW}"
ROOT="$(git rev-parse --show-toplevel)"

[[ "$NEW" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "Version must be x.y.z"; exit 1; }
git tag | grep -qx "$TAG" && { echo "Tag $TAG already exists"; exit 1; }

OLD=$(grep -m1 '^version = ' "$ROOT/pyproject.toml" | sed 's/version = "//;s/"//')
echo "Bumping $OLD -> $NEW"

sed -i "s/^version = \"${OLD}\"/version = \"${NEW}\"/" "$ROOT/pyproject.toml"

grep -q "version = \"${NEW}\"" "$ROOT/pyproject.toml" || { echo "pyproject.toml bump failed"; exit 1; }

git add "$ROOT/pyproject.toml"
git commit -m "chore: bump version to ${NEW}"
# Annotated + explicit message: repos with tag.forceSignAnnotated (or gpgSign) reject
# a lightweight `git tag`, and a non-interactive shell has no editor to supply one.
git tag -a "$TAG" -m "Release $TAG"
git push origin "$(git branch --show-current)"
git push origin "$TAG"

echo "Released: $TAG"
