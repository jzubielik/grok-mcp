#!/usr/bin/env bash
# Deploy skills from skills/ to your agent's skills directory ($SKILLS_DIR)
# Usage: SKILLS_DIR=/path/to/skills scripts/deploy-skills.sh [skill_name]
#   No args: deploys all skills
#   With arg: deploys only the named skill (e.g. grok-mcp)
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
SKILLS_SRC="$ROOT/skills"
SKILLS_DST="${SKILLS_DIR:-}"

if [[ -z "$SKILLS_DST" ]]; then
  echo "ERROR: SKILLS_DIR is not set — point it at your agent's skills directory:" >&2
  echo "  make deploy-skills SKILLS_DIR=/path/to/skills" >&2
  echo "  make deploy-skill SKILL=grok-mcp SKILLS_DIR=/path/to/skills" >&2
  exit 1
fi

if [[ ! -d "$SKILLS_SRC" ]]; then
  echo "ERROR: skills/ directory not found in repo root" >&2; exit 1
fi

deploy_skill() {
  local name="$1"
  local src="$SKILLS_SRC/$name/SKILL.md"
  local dst="$SKILLS_DST/$name/SKILL.md"

  if [[ ! -f "$src" ]]; then
    echo "WARNING: $src not found — skipping" >&2; return
  fi

  mkdir -p "$(dirname "$dst")"
  cp "$src" "$dst"
  echo "  deployed: $name"
}

if [[ $# -eq 1 ]]; then
  deploy_skill "$1"
else
  echo "Deploying skills → $SKILLS_DST"
  for dir in "$SKILLS_SRC"/*/; do
    deploy_skill "$(basename "$dir")"
  done
  echo "Done."
fi
