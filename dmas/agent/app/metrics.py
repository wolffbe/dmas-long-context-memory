from prometheus_client import Counter, Gauge, Histogram

from app.config import CFG


peer_request_duration_seconds = Histogram(
    "peer_request_duration_seconds",
    "Latency of peer /peer/ask calls including toxiproxy-injected jitter.",
    labelnames=("src", "dst"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

peer_request_failures_total = Counter(
    "peer_request_failures_total",
    "Failed /peer/ask calls.",
    labelnames=("src", "dst"),
)

ask_total = Counter(
    "ask_total",
    "Total /ask calls processed.",
    labelnames=("agent",),
)

help_decision_total = Counter(
    "help_decision_total",
    "Per-/ask whether peers were actually queried (ask) or not (skip). "
    "`skip` includes both gate-blocked and LLM-chose-not-to-ask cases — see "
    "peer_help_gate_total to disambiguate.",
    labelnames=("agent", "decision"),
)

peer_help_gate_total = Counter(
    "peer_help_gate_total",
    "Per-/ask outcome of the external latency gate that controls whether "
    "the LLM is given the ask_peers tool. `allow` = ask_peers exposed; "
    "`block` = ask_peers withheld because measured RTT exceeded the threshold.",
    labelnames=("agent", "decision"),
)

# --- toxic / threshold visibility (gauges, scraped + plotted in Grafana) ---
peer_toxic_latency_ms = Gauge(
    "peer_toxic_latency_ms",
    "Toxiproxy `latency` toxic value currently applied to this agent's peer "
    "outbound proxies (ms). 0 when no toxic is set.",
    labelnames=("src",),
)

peer_toxic_jitter_ms = Gauge(
    "peer_toxic_jitter_ms",
    "Toxiproxy `latency` toxic `jitter` attribute currently applied to this "
    "agent's peer outbound proxies (ms). 0 when no toxic is set.",
    labelnames=("src",),
)

peer_latency_threshold_ms = Gauge(
    "peer_latency_threshold_ms",
    "`peer_latency_threshold_ms` from the most recent /ask served by this "
    "agent (ms). 0 when no /ask has been received yet or the request omitted "
    "the field.",
    labelnames=("src",),
)


def src() -> str:
    return CFG.agent_id
