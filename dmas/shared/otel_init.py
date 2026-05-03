"""OTel bootstrap. Pushes spans via OTLP/HTTP to langfuse-web v3 at
`/api/public/otel/v1/traces` with HTTP Basic auth derived from
LANGFUSE_PUBLIC_KEY:LANGFUSE_SECRET_KEY. No-op when those env vars are
absent so the service still boots without observability creds.

Three step API used at app construction time:
  init(service_name)           — set TracerProvider + BatchSpanProcessor
  instrument_fastapi(app)      — server spans on every inbound request
  instrument_requests()        — client spans on `requests.post(...)`
  instrument_httpx()           — client spans on `httpx.AsyncClient(...)`

W3C `traceparent` is propagated automatically by both client
instrumentations, so coordinator → memory / responder hops stitch into
one cross-service trace without per-call wiring.
"""
from __future__ import annotations

import base64
import logging
import os

logger = logging.getLogger(__name__)

_initialised = False


def _enabled() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def init(service_name: str) -> None:
    global _initialised
    if _initialised:
        return
    _initialised = True
    if not _enabled():
        logger.info("otel disabled: LANGFUSE_PUBLIC_KEY/SECRET_KEY not set")
        return

    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    host = os.getenv("LANGFUSE_HOST", "http://langfuse-web:3000")
    endpoint = os.getenv(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        f"{host}/api/public/otel/v1/traces",
    )
    auth = base64.b64encode(
        f"{os.environ['LANGFUSE_PUBLIC_KEY']}:{os.environ['LANGFUSE_SECRET_KEY']}".encode()
    ).decode()
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(
        endpoint=endpoint,
        headers={"Authorization": f"Basic {auth}"},
    )))
    trace.set_tracer_provider(provider)
    logger.info("otel initialised: service=%s endpoint=%s", service_name, endpoint)


def instrument_fastapi(app) -> None:
    if not _enabled():
        return
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    FastAPIInstrumentor.instrument_app(app)


def instrument_requests() -> None:
    if not _enabled():
        return
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    RequestsInstrumentor().instrument()


def instrument_httpx() -> None:
    if not _enabled():
        return
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    HTTPXClientInstrumentor().instrument()
