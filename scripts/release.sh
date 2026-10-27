#!/usr/bin/env bash
# master only changes through PRs, so a release takes two steps:
#   scripts/release.sh 0.2.0        (make release VERSION=0.2.0)      bump on a branch, open a PR
#   scripts/release.sh --tag 0.2.0  (make release-tag VERSION=0.2.0)  after the merge: tag master
# Pushing the tag runs Publish (image `vX.Y.Z` + `latest`) and Release.
set -euo pipefail

usage() { echo "Usage: $0 [--tag] <version>   e.g.  $0 0.2.0" >&2; exit 1; }

MODE=bump
if [[ "${1:-}" == "--tag" ]]; then MODE=tag; shift; fi
[[ $# -eq 1 ]] || usage

NEW="$1"
TAG="v${NEW}"
ROOT="$(git rev-parse --show-toplevel)"

[[ "$NEW" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "Version must be x.y.z"; exit 1; }
git fetch -q --tags origin
git tag | grep -qx "$TAG" && { echo "Tag $TAG already exists"; exit 1; }

[[ "$(git branch --show-current)" == "master" ]] || { echo "Run from master"; exit 1; }
[[ -z "$(git status --porcelain)" ]] || { echo "Working tree is not clean"; exit 1; }
git pull -q --ff-only origin master

CURRENT=$(grep -m1 '^version = ' "$ROOT/pyproject.toml" | sed 's/version = "//;s/"//')

if [[ "$MODE" == "tag" ]]; then
  [[ "$CURRENT" == "$NEW" ]] || { echo "master is at $CURRENT, not $NEW — merge the release PR first"; exit 1; }
  # Annotated + explicit message: repos with tag.forceSignAnnotated (or gpgSign) reject
  # a lightweight `git tag`, and a non-interactive shell has no editor to supply one.
  git tag -a "$TAG" -m "Release $TAG"
  git push origin "$TAG"
  echo "Tagged: $TAG"
  exit 0
fi

echo "Bumping $CURRENT -> $NEW"
git switch -c "release/${TAG}"
sed -i "s/^version = \"${CURRENT}\"/version = \"${NEW}\"/" "$ROOT/pyproject.toml"
grep -q "version = \"${NEW}\"" "$ROOT/pyproject.toml" || { echo "pyproject.toml bump failed"; exit 1; }

git add "$ROOT/pyproject.toml"
git commit -m "chore: bump version to ${NEW}"
git push -u origin "release/${TAG}"
gh pr create --base master --title "chore: release ${TAG}" \
  --body "Bumps the version to ${NEW}. After merging: \`make release-tag VERSION=${NEW}\`."
