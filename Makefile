.PHONY: setup up down logs migrate tenant test test-integration lint e2e demo demo-site db-shell

setup:            ## create venv, install deps, seed .env
	uv sync
	@test -f .env || cp .env.example .env
	@echo "Edit .env and set GOOGLE_API_KEY (or another provider key)."

up:               ## build and start db + migrations + api + worker
	docker compose up -d --build
	@echo "API: http://localhost:$${API_PORT:-8090}/docs"

down:
	docker compose down

logs:
	docker compose logs -f api worker

tenant:           ## make tenant NAME=acme : create a tenant + API key inside the running stack (key shown once)
	docker compose exec api memory-agent tenant create $(NAME)

migrate:          ## apply migrations to DATABASE_URL (local Postgres)
	uv run memory-agent db migrate

test:             ## offline unit tests (SQLite + fake providers, no network)
	uv run pytest -q -m "not integration"

test-integration: ## needs the db container: `docker compose up -d db`
	DATABASE_URL=postgresql://memory:memory@localhost:$${POSTGRES_PORT:-5439}/memory uv run pytest -q -m integration

lint:
	uv run ruff check src tests eval scripts

e2e:              ## live end-to-end scenario against the running stack (needs a provider key)
	uv run python scripts/e2e.py

demo:
	uv run memory-agent demo

demo-site:        ## build the static recorded replay into _site (published to GitHub Pages by CI)
	rm -rf _site && mkdir _site
	cp src/memory_agent/ui/* demo/replay.js demo/fixture.json _site/
	sed 's|<script src="app.js"></script>|<script src="replay.js"></script><script src="app.js"></script>|' src/memory_agent/ui/index.html > _site/index.html
	grep -q 'replay.js' _site/index.html

db-shell:
	docker compose exec db psql -U $${POSTGRES_USER:-memory} -d $${POSTGRES_DB:-memory}
