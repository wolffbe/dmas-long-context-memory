.PHONY: start build setup shutdown load load-test experiment experiment-test render-litellm logs ps reset _bootstrap_env _start_langfuse _wait_for_agents _check_langfuse_keys

PY ?= python3

# Required on every experiment invocation — no defaults:
#   BACKEND  (mem0|zep|rag|none)
#   DATASET  (locomo|longmemeval)
#   CONV     (locomo only, e.g. 0..9)
#   QID      (longmemeval only — omit to mean "all questions")
# Optional knobs keep defaults below.
# Network shaping applied per-/ask via toxiproxy-proxy:
#   LATENCY    (ms)    — fixed latency on peer paths
#   JITTER     (ms)    — jitter range on peer paths
#   BANDWIDTH  (KB/s)  — bandwidth cap on peer paths (0 = uncapped)
LATENCY        ?= 0
JITTER         ?= 0
BANDWIDTH      ?= 0
PEER_THRESHOLD ?= 0
SEEDS          ?= 3
# Defaults: agent-1 (edge) runs local gemma; agent-2/3 (cloud) call OpenAI.
# agent-2/3 cannot route to ollama — their litellm has openai/* only.
MODEL_1   ?= gemma4:e4b
MODEL_2   ?= openai/gpt-4o-mini
MODEL_3   ?= openai/gpt-4o-mini
# Optional: override the LLM-as-judge model (defaults: mem0→gpt-5, zep→gpt-4o-mini).
JUDGE_MODEL ?=

# Agents always run with every memory backend loaded — what goes in and what
# comes out is decided per-question via request metadata, not at container
# boot. So there is no per-agent backend env var in this Makefile.
#
# Per-agent LLM model is specified ONCE — in .env (AGENT{1,2,3}_MODEL). Recipes
# below shell-source .env before rendering or building so we don't hardcode
# a model name in two places.

DMAS_COMPOSE       := dmas/docker-compose.yml
MONITORING_COMPOSE := monitoring/docker-compose.yml

# Driver scripts read every variable they need from .env (COORDINATOR_URL etc.).
# Recipes that invoke python source .env first via `set -a && . ./.env && set +a`.
RUN_ENV := set -a && . ./.env && set +a &&

# Selector derived from DATASET (locomo → --conv N, longmemeval → --qid ID or --all)
ifeq ($(DATASET),longmemeval)
  EXP_SELECTOR := $(if $(QID),--qid $(QID),--all)
else
  EXP_SELECTOR := $(if $(CONV),--conv $(CONV),--all)
endif

# Bring everything up using existing images (fast — no rebuild).
start: _bootstrap_env render-litellm _start_langfuse
	@echo "==> starting dmas stack (existing images)"
	docker compose --env-file .env -f $(DMAS_COMPOSE) up -d
	@echo "==> starting monitoring"
	docker compose --env-file .env -f $(MONITORING_COMPOSE) up -d
	@$(MAKE) _wait_for_agents
	@echo "==> ready. Grafana: http://localhost:3000   Langfuse: http://localhost:3001"

# Rebuild every image from scratch (no cache, fresh base layers). Starts langfuse
# (so litellm builds can resolve credentials), but nothing else.
build: _bootstrap_env render-litellm _start_langfuse
	@echo "==> building dmas stack from scratch (--no-cache --pull)"
	docker compose --env-file .env -f $(DMAS_COMPOSE) build --no-cache --pull
	@echo "==> build complete. Run 'make start' to bring the stack up."

_wait_for_agents:
	@echo "==> waiting for agents"
	@for p in 8011 8012 8013; do \
		for i in 1 2 3 4 5 6 7 8 9 10 11 12; do \
			curl -fs http://localhost:$$p/health >/dev/null && break; \
			sleep 2; \
		done; \
	done

# Ensure .env exists, has a usable OPENAI_API_KEY, and has Langfuse credentials.
# - If .env is missing, copy it from .env.example.
# - If OPENAI_API_KEY is blank, fail with instructions before anything else
#   runs — the openai/* litellm route, the mem0 fact-extractor, the
#   graphiti node/edge extractor, AND the LLM-as-a-judge eval all need it.
# - If LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are blank, generate random
#   ones and write them back. langfuse-web reads LANGFUSE_INIT_PROJECT_*_KEY
#   on first boot and provisions a project with exactly those values, so the
#   same keys work for every subsequent run.
_bootstrap_env:
	@if [ ! -f .env ]; then \
		echo "==> creating .env from .env.example"; \
		cp .env.example .env; \
	fi
	@OPENAI=$$(. ./.env 2>/dev/null; printf '%s' "$$OPENAI_API_KEY"); \
	if [ -z "$$OPENAI" ]; then \
		echo ""; \
		echo "ERROR: OPENAI_API_KEY is not set in .env"; \
		echo "       Edit .env and set OPENAI_API_KEY=sk-..., then re-run 'make setup'."; \
		echo "       It is used by: the openai/* litellm route (per-question LLM),"; \
		echo "       mem0 fact-extraction, graphiti node/edge extraction, and the"; \
		echo "       mem0/zep LLM-as-a-judge evaluation."; \
		echo ""; \
		exit 1; \
	fi
	@NA=$$(. ./.env 2>/dev/null; printf '%s' "$$LANGFUSE_NEXTAUTH_SECRET"); \
	SA=$$(. ./.env 2>/dev/null; printf '%s' "$$LANGFUSE_SALT"); \
	if [ -z "$$NA" ]; then \
		NA="$$(openssl rand -hex 32)"; \
		grep -q '^LANGFUSE_NEXTAUTH_SECRET=' .env \
			&& sed -i "s|^LANGFUSE_NEXTAUTH_SECRET=.*|LANGFUSE_NEXTAUTH_SECRET=$$NA|" .env \
			|| echo "LANGFUSE_NEXTAUTH_SECRET=$$NA" >> .env; \
	fi; \
	if [ -z "$$SA" ]; then \
		SA="$$(openssl rand -hex 32)"; \
		grep -q '^LANGFUSE_SALT=' .env \
			&& sed -i "s|^LANGFUSE_SALT=.*|LANGFUSE_SALT=$$SA|" .env \
			|| echo "LANGFUSE_SALT=$$SA" >> .env; \
	fi
	@# Langfuse PUBLIC/SECRET keys are NOT auto-generated. After `make start`,
	@# the user creates a project in the Langfuse UI and pastes the keys via
	@# `make setup` (load/experiment targets refuse to run until that's done).

# Bring up langfuse-postgres + clickhouse + web first and wait until /api/public/health
# is OK, so litellm-* (which depend on langfuse credentials at startup) and the agents
# (which call langfuse from request handlers) come up against a ready Langfuse.
_start_langfuse:
	@echo "==> starting langfuse stack first"
	docker compose --env-file .env -f $(DMAS_COMPOSE) up -d langfuse-postgres langfuse-clickhouse langfuse-web
	@echo "==> waiting for langfuse to become healthy"
	@for i in $$(seq 1 60); do \
		if docker exec dmas-langfuse-web node -e "fetch('http://'+require('os').hostname()+':3000/api/public/health').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))" >/dev/null 2>&1; then \
			echo "==> langfuse ready (project provisioned with keys from .env)"; \
			exit 0; \
		fi; \
		sleep 2; \
	done; \
	echo "langfuse did not become healthy within 120s"; exit 1

# Stop containers, leave volumes/network alone. Use `make reset` to also
# drop volumes + network (preserves ollama by default; `ALL=1` to drop ollama too).
# `--remove-orphans` cleans up containers from prior topologies (e.g. the
# old locomo/longmemeval/toxiproxy-proxy services).
shutdown:
	docker compose --env-file .env -f $(MONITORING_COMPOSE) down --remove-orphans || true
	docker compose --env-file .env -f $(DMAS_COMPOSE) down --remove-orphans || true

# Render dmas/litellm/agent{1,2,3}.yaml. The yaml is static
# (gemma4:e4b + openai/*), so no .env sourcing is needed.
render-litellm:
	@$(PY) dmas/litellm/render_configs.py

# One-time, after `make build` (which started Langfuse):
#   1. Open http://localhost:3001 (Langfuse UI)
#   2. Sign up, create an organization, create a project
#   3. Settings → API Keys → "Create new API keys"
#   4. Run `make setup` and paste them when prompted.
# Rebuilds litellm + agents + coordinator so the new keys are baked into those images.
# Does not start the stack — run `make start` afterwards.
setup:
	@echo "==> Open Langfuse at http://localhost:3001"
	@echo "    Sign up, create an org + project, then Settings → API Keys."
	@echo
	@printf "Public key (pk-lf-...): "; read PK; \
	 printf "Secret key (sk-lf-...): "; read SK; \
	 case "$$PK" in pk-lf-*) ;; *) echo "ERROR: public key must start with pk-lf-"; exit 2 ;; esac; \
	 case "$$SK" in sk-lf-*) ;; *) echo "ERROR: secret key must start with sk-lf-"; exit 2 ;; esac; \
	 grep -q '^LANGFUSE_PUBLIC_KEY=' .env \
		&& sed -i "s|^LANGFUSE_PUBLIC_KEY=.*|LANGFUSE_PUBLIC_KEY=$$PK|" .env \
		|| echo "LANGFUSE_PUBLIC_KEY=$$PK" >> .env; \
	 grep -q '^LANGFUSE_SECRET_KEY=' .env \
		&& sed -i "s|^LANGFUSE_SECRET_KEY=.*|LANGFUSE_SECRET_KEY=$$SK|" .env \
		|| echo "LANGFUSE_SECRET_KEY=$$SK" >> .env; \
	 echo "==> .env updated"
	@echo "==> rebuilding services that consume Langfuse keys"
	docker compose --env-file .env -f $(DMAS_COMPOSE) build --no-cache \
		litellm-1 litellm-2 litellm-3 agent-1 agent-2 agent-3 coordinator
	@echo "==> setup complete. Run 'make start' to bring the stack up."

# Internal: refuse to run if Langfuse PUBLIC/SECRET keys haven't been pasted via `make setup`.
_check_langfuse_keys:
	@PK=$$(. ./.env 2>/dev/null; printf '%s' "$$LANGFUSE_PUBLIC_KEY"); \
	 SK=$$(. ./.env 2>/dev/null; printf '%s' "$$LANGFUSE_SECRET_KEY"); \
	 if [ -z "$$PK" ] || [ -z "$$SK" ]; then \
		echo "ERROR: Langfuse keys missing in .env."; \
		echo "       Run \`make setup\` first (after \`make start\`)."; \
		exit 2; \
	 fi

# Load conversation(s) into a backend's memory store. Run ONCE per
# (BACKEND, DATASET, slug); subsequent `make experiment` calls reuse it.
#   make load BACKEND=mem0 DATASET=locomo CONV=0
#   make load BACKEND=zep  DATASET=longmemeval QID=foo
#   make load BACKEND=rag  DATASET=longmemeval        # all 500 questions
load: _check_langfuse_keys
	@case "$(BACKEND)" in mem0|zep|rag|none) ;; *) \
		echo "BACKEND is required (mem0|zep|rag|none) — pass BACKEND=..."; exit 2 ;; esac
	@case "$(DATASET)" in locomo|longmemeval) ;; *) \
		echo "DATASET is required (locomo|longmemeval) — pass DATASET=..."; exit 2 ;; esac
	$(RUN_ENV) $(PY) experiments/load.py \
		--backend $(BACKEND) --dataset $(DATASET) $(EXP_SELECTOR)

# Single entry point. Usage:
#   make experiment BACKEND=mem0 DATASET=locomo CONV=0
#   make experiment BACKEND=zep DATASET=longmemeval QID=foo LATENCY=50 JITTER=20 BANDWIDTH=512
#   make experiment BACKEND=rag DATASET=locomo CONV=0 \
#                   MODEL_1=gemma4:e4b MODEL_2=openai/gpt-4o MODEL_3=openai/gpt-4o SEEDS=3
experiment: render-litellm _check_langfuse_keys
	@case "$(BACKEND)" in mem0|zep|rag|none) ;; *) \
		echo "BACKEND is required (mem0|zep|rag|none) — pass BACKEND=..."; exit 2 ;; esac
	@case "$(DATASET)" in locomo|longmemeval) ;; *) \
		echo "DATASET is required (locomo|longmemeval) — pass DATASET=..."; exit 2 ;; esac
	$(RUN_ENV) $(PY) experiments/experiment.py \
		--backend $(BACKEND) --dataset $(DATASET) \
		--latency $(LATENCY) --jitter $(JITTER) --bandwidth $(BANDWIDTH) \
		--peer-threshold-ms $(PEER_THRESHOLD) \
		--seeds $(SEEDS) \
		--model-1 $(MODEL_1) --model-2 $(MODEL_2) --model-3 $(MODEL_3) \
		$(if $(JUDGE_MODEL),--judge-model $(JUDGE_MODEL)) \
		$(EXP_SELECTOR)

# Smoke tests — load the first turn / ask one question. Useful to verify the
# stack is wired correctly before kicking off a long run. Smoke runs prefix
# `experiment_name` with `test_` in the CSV so they're distinguishable.
#   make load-test       BACKEND=mem0 DATASET=locomo CONV=0
#   make experiment-test BACKEND=mem0 DATASET=locomo CONV=0
load-test: _check_langfuse_keys
	@case "$(BACKEND)" in mem0|zep|rag|none) ;; *) \
		echo "BACKEND is required (mem0|zep|rag|none) — pass BACKEND=..."; exit 2 ;; esac
	@case "$(DATASET)" in locomo|longmemeval) ;; *) \
		echo "DATASET is required (locomo|longmemeval) — pass DATASET=..."; exit 2 ;; esac
	$(RUN_ENV) $(PY) experiments/load.py \
		--backend $(BACKEND) --dataset $(DATASET) --limit 1 $(EXP_SELECTOR)

experiment-test: render-litellm _check_langfuse_keys
	@case "$(BACKEND)" in mem0|zep|rag|none) ;; *) \
		echo "BACKEND is required (mem0|zep|rag|none) — pass BACKEND=..."; exit 2 ;; esac
	@case "$(DATASET)" in locomo|longmemeval) ;; *) \
		echo "DATASET is required (locomo|longmemeval) — pass DATASET=..."; exit 2 ;; esac
	$(RUN_ENV) $(PY) experiments/experiment.py \
		--backend $(BACKEND) --dataset $(DATASET) \
		--latency $(LATENCY) --jitter $(JITTER) --bandwidth $(BANDWIDTH) \
		--peer-threshold-ms $(PEER_THRESHOLD) \
		--seeds 1 --limit 1 --name-prefix test_ \
		--model-1 $(MODEL_1) --model-2 $(MODEL_2) --model-3 $(MODEL_3) \
		$(if $(JUDGE_MODEL),--judge-model $(JUDGE_MODEL)) \
		$(EXP_SELECTOR)

logs:
	docker compose --env-file .env -f $(DMAS_COMPOSE) logs -f

ps:
	docker compose --env-file .env -f $(DMAS_COMPOSE) ps

# Stop everything (via shutdown) and wipe volumes + shared network. Keeps
# the ollama-data volume by default so gemma4:e4b doesn't have to be re-pulled.
# Pass ALL=1 to also drop ollama-data.
ALL ?= 0
reset: shutdown
ifeq ($(ALL),1)
	docker volume rm \
		dmas_qdrant-data-1 dmas_qdrant-data-2 dmas_qdrant-data-3 \
		dmas_neo4j-data-1 dmas_neo4j-data-2 dmas_neo4j-data-3 \
		dmas_neo4j-logs-1 dmas_neo4j-logs-2 dmas_neo4j-logs-3 \
		dmas_langfuse-postgres dmas_langfuse-clickhouse dmas_ollama-data \
		dmas_coordinator-data \
		dmas_qdrant-data dmas_neo4j-data dmas_neo4j-logs 2>/dev/null || true
else
	docker volume rm \
		dmas_qdrant-data-1 dmas_qdrant-data-2 dmas_qdrant-data-3 \
		dmas_neo4j-data-1 dmas_neo4j-data-2 dmas_neo4j-data-3 \
		dmas_neo4j-logs-1 dmas_neo4j-logs-2 dmas_neo4j-logs-3 \
		dmas_langfuse-postgres dmas_langfuse-clickhouse \
		dmas_coordinator-data \
		dmas_qdrant-data dmas_neo4j-data dmas_neo4j-logs 2>/dev/null || true
endif
	docker network rm dmas_shared-net 2>/dev/null || true

