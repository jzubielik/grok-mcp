.PHONY: hooks test lint up down verify-secrets restart logs ps release release-tag deploy-skills deploy-skill

# The configuration (secrets included) lives outside WORKSPACE_DIR / ASSETS_DIR, which the
# agent's shell can read. Override with: make up GROK_CONFIG=/path/to/other.env
GROK_CONFIG ?= $(HOME)/.config/grok-mcp/.env
# Absolute and symlink-free, resolved on the host: the server's mount check compares paths and
# cannot see through a symlink from inside the container.
GROK_CONFIG_REAL := $(shell realpath -m "$(GROK_CONFIG)")
export COMPOSE_ENV_FILES := $(GROK_CONFIG_REAL)
export GROK_ENV_FILE := $(GROK_CONFIG_REAL)

hooks:
	pdm run pre-commit install

test:
	pdm run pytest

lint:
	pdm run ruff check src tests
	pdm run ruff format --check src tests

up:
	@test -f "$(GROK_CONFIG_REAL)" || (echo "No config at $(GROK_CONFIG_REAL) — copy .env.example there (chmod 600)" && exit 1)
	@# Docker mounts a symlink's target, but the server's mount check compares the paths as
	@# written: a config under the real target would pass it. Require symlink-free mounts.
	@# One pair of surrounding quotes is stripped, as compose's env-file parser does. A `$$`
	@# is refused: compose interpolates `$${VAR}` there, so the value as written is not the path.
	@for var in WORKSPACE_DIR ASSETS_DIR; do \
		val=$$(grep "^$$var=" "$(GROK_CONFIG_REAL)" | tail -n 1 | cut -d= -f2- \
			| sed -E "s/^\"(.*)\"\$$|^'(.*)'\$$/\1\2/"); \
		[ -z "$$val" ] && continue; \
		case "$$val" in *'$$'*) \
			echo "$$var=$$val: compose would expand the \$$ in it, so it cannot be checked — use the literal absolute path"; \
			exit 1;; \
		esac; \
		real=$$(realpath -m "$$val"); \
		if [ "$$real" != "$$val" ]; then \
			echo "$$var=$$val is not a real path (symlink or not normalised): use $$real"; \
			exit 1; \
		fi; \
	done
	docker compose up -d --build

down:
	docker compose down

# A running stack must not expose a secret to the agent's uid through any /proc/<pid>/environ
# (tini as PID 1 holds Docker's whole environment). The check runs as that uid, the way `shell`
# would.
verify-secrets:
	@docker compose exec -T grok-mcp python - < scripts/verify_secrets.py

restart:
	docker compose restart grok-mcp

logs:
	docker compose logs -f --no-log-prefix grok-mcp

ps:
	docker compose ps

release:
	@test -n "$(VERSION)" || (echo "Usage: make release VERSION=0.2.0" && exit 1)
	@bash scripts/release.sh $(VERSION)

release-tag:
	@test -n "$(VERSION)" || (echo "Usage: make release-tag VERSION=0.2.0" && exit 1)
	@bash scripts/release.sh --tag $(VERSION)

deploy-skills:
	@SKILLS_DIR="$(SKILLS_DIR)" bash scripts/deploy-skills.sh

deploy-skill:
	@test -n "$(SKILL)" || (echo "Usage: make deploy-skill SKILL=grok-mcp SKILLS_DIR=/path/to/skills" && exit 1)
	@SKILLS_DIR="$(SKILLS_DIR)" bash scripts/deploy-skills.sh $(SKILL)
