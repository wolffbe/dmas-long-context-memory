.PHONY: build start stop clean reset \
        experiment experiment-leg experiment-test experiment-test-s experiment-test-l experiments logs ps \
        _check_docker _check_openai _bootstrap_env_file _refresh_public_url \
        _wait_benchmark _wait_langfuse

PY ?= python3

# Knobs for experiment / experiment-test.
# MODE: experiment-leg consumes a single value (defaults to unconstrained
# when unset). The sweep targets (experiment, experiment-test,
# experiment-test-s, experiment-test-l) accept a space-separated list and
# default to both regimes when unset. Examples:
#   make experiment-leg CONV=0 MODE=constrained
#   make experiment MODE=unconstrained
#   make experiment MODE="unconstrained constrained"
MODE           ?=
BACKENDS       ?= mem0 graphiti rag cognee full_context
LLM_AS_JUDGE_SEED ?= 3
MESSAGES       ?= $(LOAD_LIMIT)
QUESTIONS      ?= $(LIMIT)
QUESTION_TYPES ?=
Q_PER_TYPE     ?=
KEEP_STATE     ?=
NAME_PREFIX    ?=

# LOCOMO conversation indices to sweep in the full `make experiment`.
# Override per-run, e.g. `make experiment CONVS="0 5"`.
CONVS          ?= 0 1 2 3 4 5 6 7 8 9

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
	$(COMPOSE_PROJECT)_langfuse-db-data \
	$(COMPOSE_PROJECT)_langfuse-clickhouse-data \
	$(COMPOSE_PROJECT)_langfuse-clickhouse-logs \
	$(COMPOSE_PROJECT)_langfuse-minio-data

# === lifecycle ============================================================

# Build all images. Aborts if OPENAI_API_KEY is missing — every backend (mem0
# fact-extraction, graphiti node/edge extraction, judge eval, responder)
# needs it, so building without it just produces images that won't run.
build: _check_docker _bootstrap_env_file _refresh_public_url _check_openai
	@echo "==> building dmas (--no-cache --pull)"
	docker compose --env-file .env -f $(DMAS_COMPOSE) build --no-cache --pull
	@URL=$$(. ./.env 2>/dev/null; printf '%s' "$$LANGFUSE_PUBLIC_URL"); \
	echo "==> build complete. LANGFUSE_PUBLIC_URL=$$URL"; \
	echo "==> Run 'make start' next."

# Bring up the full stack headlessly. `_bootstrap_env_file` auto-
# generates the langfuse pk/sk + OTel basic-auth header into .env on
# first run; langfuse v3 picks them up via LANGFUSE_INIT_PROJECT_*.
start: _check_docker _bootstrap_env_file _check_openai
	@echo "==> starting full dmas stack"
	docker compose --env-file .env -f $(DMAS_COMPOSE) up -d --remove-orphans
	@$(MAKE) _wait_benchmark
	@echo "==> ready. Benchmark: $(BENCHMARK_URL)  Langfuse: $(LANGFUSE_URL)"

# Stop everything: kill the 'exp' tmux session (if running) so an attached
# `make experiment` is torn down cleanly, then bring the compose stack
# down. Volumes are preserved — use `clean` or `reset` to drop them.
stop:
	@echo "==> killing tmux session 'exp' (if any)"
	tmux kill-session -t exp 2>/dev/null || true
	docker compose --env-file .env -f $(DMAS_COMPOSE) down --remove-orphans || true

# Wipe ONLY memory backend state (qdrant + neo4j). Langfuse history
# and ollama models are kept intact — handy for re-running calibration
# against a fresh memory but skipping the 5-min ollama pull and the
# langfuse account setup. Compose-managed networks survive `stop` and
# are reused on the next `start`.
clean: stop
	@echo "==> dropping memory volumes: $(MEMORY_VOLUMES)"
	docker volume rm $(MEMORY_VOLUMES) 2>/dev/null || true

# Wipe everything: `stop` first (which kills the 'exp' tmux session and
# brings the compose stack down), then drop every named volume and remove
# top-level *.log plus anything under experiments/results/ that isn't in
# backup/. The three compose-managed networks (edge-net, cloud-net,
# mgmt-net) are re-created on the next `start`. Move anything worth
# keeping into experiments/results/backup/ before running.
reset: stop
	@echo "==> dropping all volumes: $(ALL_VOLUMES)"
	docker volume rm $(ALL_VOLUMES) 2>/dev/null || true
	@echo "==> removing top-level *.log"
	rm -f *.log
	@echo "==> removing experiments/results/* except backup/"
	find experiments/results -mindepth 1 -maxdepth 1 ! -name backup -exec rm -rf {} +

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
	@TOKEN=$$(curl -fsS -m 1 -X PUT 'http://169.254.169.254/latest/api/token' \
		-H 'X-aws-ec2-metadata-token-ttl-seconds: 60' 2>/dev/null || true); \
	IP=""; \
	if [ -n "$$TOKEN" ]; then \
		IP=$$(curl -fsS -m 1 -H "X-aws-ec2-metadata-token: $$TOKEN" \
			http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || true); \
	fi; \
	if [ -n "$$IP" ]; then \
		NEW_URL="http://$$IP:3000"; \
		OLD_URL=$$(. ./.env 2>/dev/null; printf '%s' "$$LANGFUSE_PUBLIC_URL"); \
		grep -v '^LANGFUSE_PUBLIC_URL=' .env > .env.tmp; mv .env.tmp .env; \
		echo "LANGFUSE_PUBLIC_URL=$$NEW_URL" >> .env; \
		if [ "$$OLD_URL" != "$$NEW_URL" ]; then \
			echo "==> refreshed EC2 public IP: LANGFUSE_PUBLIC_URL=$$NEW_URL (was: $${OLD_URL:-unset})"; \
		fi; \
	elif ! grep -q '^LANGFUSE_PUBLIC_URL=http' .env; then \
		echo "LANGFUSE_PUBLIC_URL=http://localhost:3000" >> .env; \
		echo "==> no EC2 metadata; LANGFUSE_PUBLIC_URL defaulted to localhost"; \
	fi
	@if ! grep -q '^LANGFUSE_PUBLIC_KEY=pk-lf-' .env; then \
		val=pk-lf-$$(openssl rand -hex 16); \
		grep -v '^LANGFUSE_PUBLIC_KEY=' .env > .env.tmp; mv .env.tmp .env; \
		echo "LANGFUSE_PUBLIC_KEY=$$val" >> .env; \
		echo "==> generated LANGFUSE_PUBLIC_KEY (headless bootstrap)"; \
	fi
	@if ! grep -q '^LANGFUSE_SECRET_KEY=sk-lf-' .env; then \
		val=sk-lf-$$(openssl rand -hex 24); \
		grep -v '^LANGFUSE_SECRET_KEY=' .env > .env.tmp; mv .env.tmp .env; \
		echo "LANGFUSE_SECRET_KEY=$$val" >> .env; \
		echo "==> generated LANGFUSE_SECRET_KEY (headless bootstrap)"; \
	fi
	@PK=$$(. ./.env 2>/dev/null; printf '%s' "$$LANGFUSE_PUBLIC_KEY"); \
	SK=$$(. ./.env 2>/dev/null; printf '%s' "$$LANGFUSE_SECRET_KEY"); \
	AUTH=$$(printf '%s:%s' "$$PK" "$$SK" | base64 -w0 2>/dev/null || printf '%s:%s' "$$PK" "$$SK" | base64); \
	grep -v '^LANGFUSE_OTEL_BASIC_AUTH=' .env > .env.tmp; mv .env.tmp .env; \
	echo "LANGFUSE_OTEL_BASIC_AUTH=$$AUTH" >> .env; \
	echo "==> wrote LANGFUSE_OTEL_BASIC_AUTH (litellm OTel exporter)"
	@# === pick non-colliding subnets =====================================
	@# Runs only when ANY of the five subnet/IP vars is missing — once
	@# all five are written the picker leaves them alone, so a running
	@# deployment never silently moves to a different /16 mid-life. The
	@# search compares each candidate /16's first three /24s against
	@# (a) every existing Docker network's IPAM subnet and (b) the
	@# host's IPv4 route table. `ip` is best-effort: on Git Bash /
	@# Docker Desktop where iproute2 is absent the host-route check
	@# degrades to empty, so picker relies solely on Docker state.
	@if ! grep -qE '^EDGE_NET_SUBNET=.+' .env \
		|| ! grep -qE '^EDGE_NET_TOXIPROXY_IP=.+' .env \
		|| ! grep -qE '^CLOUD_NET_SUBNET=.+' .env \
		|| ! grep -qE '^CLOUD_NET_TOXIPROXY_IP=.+' .env \
		|| ! grep -qE '^MGMT_NET_SUBNET=.+' .env; then \
		echo "==> picking non-colliding subnets"; \
		docker_subnets=$$(docker network ls -q 2>/dev/null | while read net; do \
			docker network inspect -f '{{range .IPAM.Config}}{{.Subnet}} {{end}}' $$net 2>/dev/null; \
		done); \
		host_routes=$$(ip -4 route show 2>/dev/null | awk '{print $$1}' | grep -E '^[0-9]' || true); \
		chosen=""; \
		for base in 172.30 172.40 172.50 172.60 172.70 10.42 10.52 10.62 192.168.220 192.168.230; do \
			collision=false; \
			if echo "$$docker_subnets" | grep -qE "(^| )$$base\." \
				|| echo "$$host_routes" | grep -qE "(^| )$$base\."; then \
				collision=true; \
			fi; \
			if [ "$$collision" = "false" ]; then chosen=$$base; break; fi; \
		done; \
		if [ -z "$$chosen" ]; then \
			echo "ERROR: every candidate /16 collides with an existing"; \
			echo "       Docker network or host route. Pin EDGE_NET_SUBNET"; \
			echo "       etc. manually in .env to a known-free range."; \
			exit 1; \
		fi; \
		echo "==> chose $$chosen.0.0/16 (edge=$$chosen.1.0/24 cloud=$$chosen.2.0/24 mgmt=$$chosen.3.0/24)"; \
		grep -v -E '^(EDGE_NET_SUBNET|EDGE_NET_TOXIPROXY_IP|CLOUD_NET_SUBNET|CLOUD_NET_TOXIPROXY_IP|MGMT_NET_SUBNET)=' .env > .env.tmp; \
		mv .env.tmp .env; \
		{ \
			echo "EDGE_NET_SUBNET=$$chosen.1.0/24"; \
			echo "EDGE_NET_TOXIPROXY_IP=$$chosen.1.10"; \
			echo "CLOUD_NET_SUBNET=$$chosen.2.0/24"; \
			echo "CLOUD_NET_TOXIPROXY_IP=$$chosen.2.10"; \
			echo "MGMT_NET_SUBNET=$$chosen.3.0/24"; \
		} >> .env; \
	fi

# Force-refresh LANGFUSE_PUBLIC_URL from EC2 metadata. Runs in addition
# to the IP block inside _bootstrap_env_file so `make build` always
# rewrites the URL even if the file already had a value, and so the
# behavior is independent of bootstrap's other side-effects (key
# generation etc.). On non-EC2 hosts metadata times out in 1s and the
# previous value is kept.
_refresh_public_url:
	@TOKEN=$$(curl -fsS -m 1 -X PUT 'http://169.254.169.254/latest/api/token' \
		-H 'X-aws-ec2-metadata-token-ttl-seconds: 60' 2>/dev/null || true); \
	IP=""; \
	if [ -n "$$TOKEN" ]; then \
		IP=$$(curl -fsS -m 1 -H "X-aws-ec2-metadata-token: $$TOKEN" \
			http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || true); \
	fi; \
	if [ -n "$$IP" ]; then \
		NEW_URL="http://$$IP:3000"; \
		OLD_URL=$$(. ./.env 2>/dev/null; printf '%s' "$$LANGFUSE_PUBLIC_URL"); \
		grep -v '^LANGFUSE_PUBLIC_URL=' .env > .env.tmp; mv .env.tmp .env; \
		echo "LANGFUSE_PUBLIC_URL=$$NEW_URL" >> .env; \
		if [ "$$OLD_URL" != "$$NEW_URL" ]; then \
			echo "==> refreshed EC2 public IP: LANGFUSE_PUBLIC_URL=$$NEW_URL (was: $${OLD_URL:-unset})"; \
		else \
			echo "==> LANGFUSE_PUBLIC_URL already current: $$NEW_URL"; \
		fi; \
	else \
		CURR=$$(. ./.env 2>/dev/null; printf '%s' "$$LANGFUSE_PUBLIC_URL"); \
		echo "==> EC2 metadata unavailable; keeping LANGFUSE_PUBLIC_URL=$${CURR:-unset}"; \
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

# Single-leg primitive used by the sweeps below. For each backend in
# BACKENDS the benchmark wipes memory, applies toxics for MODE, loads
# CONV, asks every question ONCE, judges every answer
# LLM_AS_JUDGE_SEED times and majority-votes, then wipes again.
# Streams NDJSON; rows land in
# experiments/results/{prefix}{backend}_{mode}.csv (one file per
# (framework, mode) — convs share a file, distinguished by the
# `conversation_index` column and `experiment_name`).
#   make experiment-leg CONV=0 MODE=unconstrained
#   make experiment-leg CONV=0 MODE=constrained LLM_AS_JUDGE_SEED=3 BACKENDS="mem0 graphiti"
#   make experiment-leg CONV=0 MESSAGES=20 QUESTIONS=1   # load 20 msgs, ask 1 q
experiment-leg:
	@if [ -z "$(CONV)" ]; then echo "CONV is required — pass CONV=<i>"; exit 2; fi
	@MODE_VAL="$(or $(MODE),unconstrained)"; \
	case "$$MODE_VAL" in unconstrained|constrained) ;; *) \
		echo "MODE must be a single value (unconstrained|constrained) — got '$$MODE_VAL'"; exit 2 ;; esac; \
	prefix="$(NAME_PREFIX)"; \
	conflict=""; \
	for backend in $(BACKENDS); do \
		csv="experiments/results/$${prefix}$${backend}_$${MODE_VAL}.csv"; \
		if [ -f "$$csv" ] && awk -F',' \
			'NR==1{for(i=1;i<=NF;i++)if($$i=="conversation_index")col=i} \
			 NR>1&&col&&$$col=="$(CONV)"{found=1;exit} \
			 END{exit !found}' "$$csv" 2>/dev/null; then \
			conflict="$$conflict $$csv"; \
		fi; \
	done; \
	if [ -n "$$conflict" ]; then \
		if [ -n "$(FORCE)" ]; then \
			echo "WARNING: conv=$(CONV) already in:$$conflict  Proceeding (FORCE=1)."; \
		else \
			echo "ERROR: conv=$(CONV) already in:"; \
			for f in $$conflict; do echo "  $$f"; done; \
			echo "  Delete those files or pass FORCE=1 to proceed."; \
			exit 1; \
		fi; \
	fi
	@MODE_VAL="$(or $(MODE),unconstrained)"; \
	LOG_PID=""; \
	trap 'pkill -P $$LOG_PID 2>/dev/null; kill $$LOG_PID 2>/dev/null; wait $$LOG_PID 2>/dev/null; true' EXIT INT TERM; \
	( docker compose --env-file .env -f $(DMAS_COMPOSE) logs -f --tail=0 memory benchmark 2>&1 \
		| grep -E --line-buffered '\[(mem0|graphiti|cognee|rag) load\]|FullContext append|\[load\]|\[exp\] ' \
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
	body=$$(printf '{"conv":%s,"mode":"%s","llm_as_judge_seed":%s,"backends":[%s]%s}' \
		"$(CONV)" "$$MODE_VAL" "$(LLM_AS_JUDGE_SEED)" "$$backends_json" "$$extras"); \
	echo "==> POST $(BENCHMARK_URL)/experiment  body=$$body"; \
	curl -fsSN -X POST "$(BENCHMARK_URL)/experiment" \
		-H "Content-Type: application/json" \
		-d "$$body"; \
	sleep 0.5

# Calibration-as-test: load 119 messages and ask the first 3 questions
# per category in 1-4 across BOTH network regimes (unconstrained +
# constrained). 119 is hand-picked: it's the smallest CONV=0 prefix
# that covers the evidence for the first three questions in each
# non-adversarial category. KEEP_STATE means each leg's post-reset is
# skipped, so subsequent invocations of the same (backend, mode) re-ask
# without reloading — iteration on retrieval / responder / judge is fast.
# Smoke defaults: LLM_AS_JUDGE_SEED=1 (one judge call per answer for
# speed; the full bench uses 3) and Q_PER_TYPE=3.
#   make experiment-test CONV=0
#   make experiment-test CONV=0 BACKENDS="mem0"
TEST_MESSAGES           ?= 119
TEST_QUESTION_TYPES     ?= 1,2,3,4
TEST_Q_PER_TYPE         ?= 3
TEST_NAME_PREFIX        ?= test_
TEST_LLM_AS_JUDGE_SEED  ?= 1
experiment-test:
	@if [ -z "$(CONV)" ]; then echo "CONV is required — pass CONV=<i>"; exit 2; fi
	@MODE_LIST="$(or $(MODE),unconstrained constrained)"; \
	for mode in $$MODE_LIST; do \
		case "$$mode" in unconstrained|constrained) ;; *) \
			echo "MODE values must be unconstrained|constrained — got '$$mode'"; exit 2 ;; esac; \
		echo "==> smoke $$mode leg (CONV=$(CONV) BACKENDS='$(BACKENDS)' LLM_AS_JUDGE_SEED=$(TEST_LLM_AS_JUDGE_SEED))"; \
		$(MAKE) experiment-leg \
			CONV=$(CONV) MODE=$$mode LLM_AS_JUDGE_SEED=$(TEST_LLM_AS_JUDGE_SEED) \
			MESSAGES=$(TEST_MESSAGES) \
			QUESTION_TYPES=$(TEST_QUESTION_TYPES) \
			Q_PER_TYPE=$(TEST_Q_PER_TYPE) \
			KEEP_STATE=1 \
			NAME_PREFIX=$(TEST_NAME_PREFIX) \
			FORCE=1 \
			BACKENDS="$(BACKENDS)" || exit $$?; \
	done

# Full publishable sweep: every CONV in CONVS × every mode in MODE ×
# every backend in BACKENDS. MODE defaults to both regimes
# (unconstrained constrained) when unset; pass a single value to narrow.
# Each (conv, mode) leg wipes memory state at the start, loads the
# conversation in full, asks every question once, and lets the
# LLM-as-judge panel run LLM_AS_JUDGE_SEED times per answer (default 3,
# majority-vote). No KEEP_STATE — every leg starts on clean state.
#
# Auto-wraps in a tmux session named 'exp' on first invocation: SIGINT is
# trapped so Ctrl+C inside the pane is a no-op, and stdout is teed to
# exp_<backends-joined-by-underscore>.log. Detach with Ctrl-b d; re-run
# the same command (or `tmux attach -t exp`) to re-attach. `make stop`
# kills the session. Already inside tmux ($$TMUX set) → skips the wrap
# and runs the sweep directly.
#
# Override CONVS, MODE, or BACKENDS to narrow:
#   make experiment
#   make experiment BACKENDS="mem0 graphiti"
#   make experiment CONVS="0 5"
#   make experiment MODE=unconstrained
#   make experiment LLM_AS_JUDGE_SEED=5
experiment:
	@if [ -z "$$TMUX" ]; then \
		LOG_FILE="exp_$$(echo '$(BACKENDS)' | tr ' ' '_').log"; \
		echo "==> launching tmux session 'exp' (log: $$LOG_FILE)"; \
		exec tmux new -A -s exp "bash -c 'trap \"\" INT; $(MAKE) experiment $(MAKEOVERRIDES) 2>&1 | tee -a $$LOG_FILE'"; \
	fi; \
	MODE_LIST="$(or $(MODE),unconstrained constrained)"; \
	for conv in $(CONVS); do \
		for mode in $$MODE_LIST; do \
			case "$$mode" in unconstrained|constrained) ;; *) \
				echo "MODE values must be unconstrained|constrained — got '$$mode'"; exit 2 ;; esac; \
			echo "==> experiment leg conv=$$conv mode=$$mode (BACKENDS='$(BACKENDS)' LLM_AS_JUDGE_SEED=$(LLM_AS_JUDGE_SEED))"; \
			$(MAKE) experiment-leg \
				CONV=$$conv MODE=$$mode \
				LLM_AS_JUDGE_SEED=$(LLM_AS_JUDGE_SEED) \
				BACKENDS="$(BACKENDS)" || exit $$?; \
		done; \
	done

# Short smoke: load the first 5 messages of CONV (default 0), ask the
# first question of LoCoMo category 2 (multi-hop), across BOTH regimes,
# across every backend in BACKENDS. Single judge call per answer to keep
# wall time low. KEEP_STATE so the constrained leg reuses the
# unconstrained leg's load.
#   make experiment-test-s
#   make experiment-test-s CONV=3
TEST_S_CONV          ?= 0
TEST_S_MESSAGES      ?= 5
TEST_S_QUESTION_TYPES ?= 2
TEST_S_Q_PER_TYPE    ?= 1
TEST_S_NAME_PREFIX   ?= test_s_
# Smoke runs default to a single judge call for speed. A command-line
# override (`make experiment-test-s LLM_AS_JUDGE_SEED=3`) wins via
# `$(origin)`, but the top-level default of 3 doesn't leak into the smoke.
TEST_S_LLM_AS_JUDGE_SEED ?= 1
experiment-test-s:
	@CONV_VAL="$(or $(CONV),$(TEST_S_CONV))"; \
	SEED="$(if $(filter command,$(origin LLM_AS_JUDGE_SEED)),$(LLM_AS_JUDGE_SEED),$(TEST_S_LLM_AS_JUDGE_SEED))"; \
	MODE_LIST="$(or $(MODE),unconstrained constrained)"; \
	for mode in $$MODE_LIST; do \
		case "$$mode" in unconstrained|constrained) ;; *) \
			echo "MODE values must be unconstrained|constrained — got '$$mode'"; exit 2 ;; esac; \
		echo "==> smoke-S $$mode leg (CONV=$$CONV_VAL BACKENDS='$(BACKENDS)' MESSAGES=$(TEST_S_MESSAGES) QT=$(TEST_S_QUESTION_TYPES) Q_PER_TYPE=$(TEST_S_Q_PER_TYPE) LLM_AS_JUDGE_SEED=$$SEED)"; \
		$(MAKE) experiment-leg \
			CONV=$$CONV_VAL MODE=$$mode \
			LLM_AS_JUDGE_SEED=$$SEED \
			MESSAGES=$(TEST_S_MESSAGES) \
			QUESTION_TYPES=$(TEST_S_QUESTION_TYPES) \
			Q_PER_TYPE=$(TEST_S_Q_PER_TYPE) \
			KEEP_STATE=1 \
			NAME_PREFIX=$(TEST_S_NAME_PREFIX) \
			FORCE=1 \
			BACKENDS="$(BACKENDS)" || exit $$?; \
	done

# Long smoke: load the first 199 messages of CONV (default 0), ask the
# first 3 questions per LoCoMo category 1-4 (single-hop, multi-hop,
# temporal, open-domain), across BOTH regimes, across every backend in
# BACKENDS. Single judge call per answer; KEEP_STATE so the constrained
# leg reuses the unconstrained load.
#   make experiment-test-l
#   make experiment-test-l CONV=2 BACKENDS="mem0 graphiti"
TEST_L_CONV          ?= 0
TEST_L_MESSAGES      ?= 199
TEST_L_QUESTION_TYPES ?= 1,2,3,4
TEST_L_Q_PER_TYPE    ?= 3
TEST_L_NAME_PREFIX   ?= test_l_
TEST_L_LLM_AS_JUDGE_SEED ?= 1
experiment-test-l:
	@CONV_VAL="$(or $(CONV),$(TEST_L_CONV))"; \
	SEED="$(if $(filter command,$(origin LLM_AS_JUDGE_SEED)),$(LLM_AS_JUDGE_SEED),$(TEST_L_LLM_AS_JUDGE_SEED))"; \
	MODE_LIST="$(or $(MODE),unconstrained constrained)"; \
	for mode in $$MODE_LIST; do \
		case "$$mode" in unconstrained|constrained) ;; *) \
			echo "MODE values must be unconstrained|constrained — got '$$mode'"; exit 2 ;; esac; \
		echo "==> smoke-L $$mode leg (CONV=$$CONV_VAL BACKENDS='$(BACKENDS)' MESSAGES=$(TEST_L_MESSAGES) QT=$(TEST_L_QUESTION_TYPES) Q_PER_TYPE=$(TEST_L_Q_PER_TYPE) LLM_AS_JUDGE_SEED=$$SEED)"; \
		$(MAKE) experiment-leg \
			CONV=$$CONV_VAL MODE=$$mode \
			LLM_AS_JUDGE_SEED=$$SEED \
			MESSAGES=$(TEST_L_MESSAGES) \
			QUESTION_TYPES=$(TEST_L_QUESTION_TYPES) \
			Q_PER_TYPE=$(TEST_L_Q_PER_TYPE) \
			KEEP_STATE=1 \
			NAME_PREFIX=$(TEST_L_NAME_PREFIX) \
			BACKENDS="$(BACKENDS)" || exit $$?; \
	done

# Back-compat alias — `make experiments` (plural) used to dispatch the
# bash sweeper; both now reduce to the in-Makefile sweep above.
experiments: experiment

logs:
	docker compose --env-file .env -f $(DMAS_COMPOSE) logs -f

ps:
	docker ps
