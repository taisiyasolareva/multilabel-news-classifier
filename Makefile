## Repo convenience commands (reviewer-friendly)

.PHONY: help demo up down logs status rebuild rebuild-api clean

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

demo: ## Run the full demo (API + 4 dashboards) in foreground
	docker compose up --build

up: ## Start the stack in background
	docker compose up -d

down: ## Stop and remove containers/networks
	docker compose down

logs: ## Tail API logs (most useful for debugging)
	docker compose logs -f api

status: ## Show container status and ports
	docker compose ps

rebuild: ## Rebuild and restart everything (background)
	docker compose down
	docker compose up --build -d

rebuild-api: ## Rebuild API image without cache (fixes “cached Dockerfile” surprises)
	docker compose build --no-cache api
	docker compose up -d --force-recreate api

clean: ## Remove containers + volumes (nuclear reset)
	docker compose down -v

