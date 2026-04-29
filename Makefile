.PHONY: build setup start stop clean reset \
        experiment experiment-test experiments logs ps \
        _check_docker _check_openai _check_langfuse_keys _bootstrap_env_file \
        _wait_benchmark _wait_langfuse

PY ?= python3

# Knobs for experiment / experiment-test.
MODE           ?= unconstrained
BACKENDS       ?= mem0 graphiti rag cognee full_context
SEEDS          ?= 3
MESSAGES       ?= $(LOAD_LIMIT)
QUESTIONS      ?= $(LIMIT)
QUESTION_TYPES ?=
Q_PER_TYPE     ?=
KEEP_STATE     ?=
NAME_PREFIX    ?=
ALL            ?= 0

DMAS_COMPOSE  := dmas/docker-compose.yml
BENCHMARK_URL ?= http://localhost:8002
LANGFUSE_URL  ?= http://localhost:3000

# Compose volume names (project prefix is the dir name normalised by compose).
COMPOSE_PROJECT := dmas-long-context-memory
MEMORY_VOLUMES  := \
	$(COMPOSE_PROJECT)_qdrant-data \
	$(COMPOSE_PROJECT)_neo4j-data \
	$(COMPOSE_PROJECT)_neo4j-logs
ALL_VOLUMES := \
	$(MEMORY_VOLUMES) \
	$(COMPOSE_PROJECT)_ollama_data \
	$(COMPOSE_PROJECT)_langfuse-db-data

# === lifecycle ============================================================

# Build all images. Aborts if OPENAI_API_KEY is missing — every backend (mem0
# fact-extraction, graphiti node/edge extraction, judge eval, responder)
# needs it, so building without it just produces images that won't run.
build: _check_docker _bootstrap_env_file _check_openai
	@echo "==> building dmas (--no-cache --pull)"
	docker compose --env-file .env -f $(DMAS_COMPOSE) build --no-cache --pull
	@echo "==> build complete. Run 'make setup' next."

# One-time bootstrap: bring Langfuse up alone, prompt for API keys, persist
# them to .env. After this, `make start` brings the rest of the stack up
# with traces wired through litellm. Subsequent runs only need `make start`.
setup: _check_docker _bootstrap_env_file
	@docker network create dmas-network 2>/dev/null || true
	@echo "==> starting langfuse..."
	docker compose --env-file .env -f $(DMAS_COMPOSE) up -d langfuse-db langfuse-web
	@$(MAKE) _wait_langfuse
	@PUBLIC_URL=$$(. ./.env 2>/dev/null; printf '%s' "$${LANGFUSE_PUBLIC_URL:-$(LANGFUSE_URL)}"); \
	echo ""; \
	echo "==> Langfuse is up at $$PUBLIC_URL"; \
	echo ""; \
	echo "  1. Open it in a browser"; \
	echo "  2. Sign in (default: dev@local.dev / devdevdev) or sign up"; \
	echo "  3. Create or open a project, then Settings -> API Keys"; \
	echo "  4. Generate a keypair and paste below"; \
	echo ""; \
	printf 'LANGFUSE_PUBLIC_KEY (pk-lf-...): '; read pk; \
	printf 'LANGFUSE_SECRET_KEY (sk-lf-...): '; read sk; \
	if [ -z "$$pk" ] || [ -z "$$sk" ]; then \
		echo "ERROR: empty key — aborting without writing .env"; exit 2; \
	fi; \
	case "$$pk" in pk-lf-*) ;; *) echo "ERROR: public key must start with pk-lf-"; exit 2 ;; esac; \
	case "$$sk" in sk-lf-*) ;; *) echo "ERROR: secret key must start with sk-lf-"; exit 2 ;; esac; \
	tmp=$$(mktemp); \
	grep -v -E '^LANGFUSE_(PUBLIC|SECRET)_KEY=' .env > $$tmp; \
	printf 'LANGFUSE_PUBLIC_KEY=%s\nLANGFUSE_SECRET_KEY=%s\n' "$$pk" "$$sk" >> $$tmp; \
	mv $$tmp .env; \
	echo ""; \
	echo "==> wrote LANGFUSE_PUBLIC_KEY/SECRET_KEY to .env"; \
	echo "==> run 'make start' to bring up the rest of the stack"

# Bring up the full stack. Assumes `make setup` has been run once so the
# langfuse keys are in .env — `_check_langfuse_keys` aborts otherwise.
start: _check_docker _bootstrap_env_file _check_openai _check_langfuse_keys
	@docker network create dmas-network 2>/dev/null || true
	@echo "==> starting full dmas stack"
	docker compose --env-file .env -f $(DMAS_COMPOSE) up -d --remove-orphans
	@$(MAKE) _wait_benchmark
	@echo "==> ready. Benchmark: $(BENCHMARK_URL)  Langfuse: $(LANGFUSE_URL)"

# Stop everything. Volumes are preserved — use `clean` or `reset` to drop them.
stop:
	docker compose --env-file .env -f $(DMAS_COMPOSE) down --remove-orphans || true

# Wipe ONLY memory backend state (qdrant + neo4j). Langfuse history,
# ollama models, and the dmas-network are kept intact — handy for
# re-running calibration against a fresh memory but skipping the
# 5-min ollama pull and the langfuse account setup.
clean: stop
	@echo "==> dropping memory volumes: $(MEMORY_VOLUMES)"
	docker volume rm $(MEMORY_VOLUMES) 2>/dev/null || true

# Wipe everything: every named volume + the dmas-network. Next `make
# setup` starts from a blank slate.
reset: stop
	@echo "==> dropping all volumes: $(ALL_VOLUMES)"
	docker volume rm $(ALL_VOLUMES) 2>/dev/null || true
	docker network rm dmas-network 2>/dev/null || true

# === checks ==============================================================

_check_docker:
	@if ! command -v docker >/dev/null 2>&1; then \
		echo "ERROR: docker is not installed or not on PATH."; \
		echo "       Install Docker Desktop or Docker Engine: https://docs.docker.com/get-docker/"; \
		exit 1; \
	fi
	@if ! docker compose version >/dev/null 2>&1; then \
		echo "ERROR: 'docker compose' plugin is not available."; \
		echo "       Install the Compose v2 plugin: https://docs.docker.com/compose/install/"; \
		exit 1; \
	fi
	@if ! docker info >/dev/null 2>&1; then \
		echo "ERROR: docker daemon is not running or current user cannot reach it."; \
		echo "       Start Docker, or add your user to the 'docker' group."; \
		exit 1; \
	fi

_bootstrap_env_file:
	@if [ ! -f .env ]; then \
		echo "==> creating .env from .env.example"; \
		cp .env.example .env; \
	fi
	@for var in LANGFUSE_NEXTAUTH_SECRET LANGFUSE_SALT; do \
		if ! grep -q "^$$var=." .env; then \
			val=$$(openssl rand -hex 32); \
			grep -v "^$$var=" .env > .env.tmp; mv .env.tmp .env; \
			echo "$$var=$$val" >> .env; \
			echo "==> generated $$var"; \
		fi; \
	done
	@if ! grep -q '^LANGFUSE_ENCRYPTION_KEY=.\{64,\}' .env; then \
		val=$$(openssl rand -hex 32); \
		grep -v '^LANGFUSE_ENCRYPTION_KEY=' .env > .env.tmp; mv .env.tmp .env; \
		echo "LANGFUSE_ENCRYPTION_KEY=$$val" >> .env; \
		echo "==> generated LANGFUSE_ENCRYPTION_KEY (256-bit hex)"; \
	fi

_check_openai:
	@OPENAI=$$(. ./.env 2>/dev/null; printf '%s' "$$OPENAI_API_KEY"); \
	if [ -z "$$OPENAI" ]; then \
		echo ""; \
		echo "ERROR: OPENAI_API_KEY is not set in .env"; \
		echo "       Used by: litellm openai/* route, mem0 fact-extraction,"; \
		echo "       graphiti node/edge extraction, responder, and the judge."; \
		echo "       Add a line to .env: OPENAI_API_KEY=sk-..."; \
		echo ""; \
		exit 1; \
	fi

_check_langfuse_keys:
	@PK=$$(. ./.env 2>/dev/null; printf '%s' "$$LANGFUSE_PUBLIC_KEY"); \
	SK=$$(. ./.env 2>/dev/null; printf '%s' "$$LANGFUSE_SECRET_KEY"); \
	if [ -z "$$PK" ] || [ -z "$$SK" ]; then \
		echo ""; \
		echo "ERROR: LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY missing from .env"; \
		echo "       Run 'make setup' first to bring Langfuse up and capture the keys."; \
		echo ""; \
		exit 1; \
	fi

_wait_langfuse:
	@echo "==> waiting for langfuse"
	@for i in $$(seq 1 30); do \
		curl -fs $(LANGFUSE_URL)/api/public/health >/dev/null 2>&1 && exit 0; \
		sleep 2; \
	done; \
	echo "langfuse did not become ready"; exit 1

_wait_benchmark:
	@echo "==> waiting for benchmark"
	@for i in $$(seq 1 12); do \
		curl -fs $(BENCHMARK_URL)/health >/dev/null && exit 0; \
		sleep 2; \
	done; \
	echo "benchmark did not become ready"; exit 1

# === workloads ===========================================================

# Single operator entrypoint. For each backend in BACKENDS the benchmark
# wipes memory, applies toxics for MODE, loads CONV, asks every question for
# SEEDS independent runs, then wipes again. Streams NDJSON; rows land in
# experiments/results/results.csv.
#   make experiment CONV=0 MODE=unconstrained
#   make experiment CONV=0 MODE=constrained SEEDS=3 BACKENDS="mem0 graphiti"
#   make experiment CONV=0 MESSAGES=20 QUESTIONS=1   # load 20 msgs, ask 1 q
experiment:
	@if [ -z "$(CONV)" ]; then echo "CONV is required — pass CONV=<i>"; exit 2; fi
	@case "$(MODE)" in unconstrained|constrained) ;; *) \
		echo "MODE must be unconstrained|constrained — got '$(MODE)'"; exit 2 ;; esac
	@LOG_PID=""; \
	trap 'kill $$LOG_PID 2>/dev/null; wait $$LOG_PID 2>/dev/null; true' EXIT INT TERM; \
	( docker compose --env-file .env -f $(DMAS_COMPOSE) logs -f --tail=0 memory benchmark 2>&1 \
		| grep -E --line-buffered '\[(mem0|graphiti) load\]|\[load\]|\[exp\] ' \
		| sed -u 's/^/[svc] /' ) & LOG_PID=$$!; \
	sleep 0.3; \
	extras=""; \
	if [ -n "$(QUESTIONS)" ]; then extras="$$extras,\"limit\":$(QUESTIONS)"; fi; \
	if [ -n "$(MESSAGES)" ]; then extras="$$extras,\"load_limit\":$(MESSAGES)"; fi; \
	if [ -n "$(QUESTION_TYPES)" ]; then \
		qt_json=$$(printf '%s' "$(QUESTION_TYPES)" | sed -E 's/[[:space:]]+//g; s/^,+|,+$$//g'); \
		extras="$$extras,\"question_types\":[$$qt_json]"; \
	fi; \
	if [ -n "$(Q_PER_TYPE)" ]; then extras="$$extras,\"q_per_type\":$(Q_PER_TYPE)"; fi; \
	if [ -n "$(KEEP_STATE)" ]; then extras="$$extras,\"skip_post_reset\":true"; fi; \
	if [ -n "$(NAME_PREFIX)" ]; then extras="$$extras,\"name_prefix\":\"$(NAME_PREFIX)\""; fi; \
	backends_json=$$(printf '"%s",' $(BACKENDS) | sed 's/,$$//'); \
	body=$$(printf '{"conv":%s,"mode":"%s","seeds":%s,"backends":[%s]%s}' \
		"$(CONV)" "$(MODE)" "$(SEEDS)" "$$backends_json" "$$extras"); \
	echo "==> POST $(BENCHMARK_URL)/experiment  body=$$body"; \
	curl -fsSN -X POST "$(BENCHMARK_URL)/experiment" \
		-H "Content-Type: application/json" \
		-d "$$body"; \
	sleep 0.5

# Calibration-as-test: load 119 messages and ask the first 3 questions
# per category in 1-4, single seed, unconstrained, keep memory state.
# 119 is hand-picked: it's the smallest CONV=0 prefix that covers the
# evidence for the first three questions in each non-adversarial
# category. The benchmark does not auto-derive that — the operator
# specifies it. KEEP_STATE means subsequent runs skip the load and only
# re-ask, so iteration on retrieval / responder / judge is fast.
#   make experiment-test CONV=0
#   make experiment-test CONV=0 BACKENDS="mem0"
TEST_MESSAGES        ?= 119
TEST_QUESTION_TYPES  ?= 1,2,3,4
TEST_Q_PER_TYPE      ?= 3
TEST_NAME_PREFIX     ?= test_
experiment-test:
	@if [ -z "$(CONV)" ]; then echo "CONV is required — pass CONV=<i>"; exit 2; fi
	@$(MAKE) experiment \
		CONV=$(CONV) MODE=unconstrained SEEDS=1 \
		MESSAGES=$(TEST_MESSAGES) \
		QUESTION_TYPES=$(TEST_QUESTION_TYPES) \
		Q_PER_TYPE=$(TEST_Q_PER_TYPE) \
		KEEP_STATE=1 \
		NAME_PREFIX=$(TEST_NAME_PREFIX) \
		BACKENDS="$(BACKENDS)"

# Full sweep: both modes for one CONV. Wipes happen automatically per backend
# inside the benchmark, so unconstrained and constrained never share state.
experiments:
	bash experiments/experiments.sh

logs:
	docker compose --env-file .env -f $(DMAS_COMPOSE) logs -f

ps:
	docker ps
