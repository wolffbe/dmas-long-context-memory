import os
import threading

import uvicorn
from fastapi import FastAPI
from prometheus_client import start_http_server

from app.llm_accounting import install as _install_llm_accounting

# Patch openai SDK BEFORE any module instantiates a client (mem0/graphiti
# create their internals lazily, but `app.llm` builds an AsyncOpenAI at import
# time — so install before importing it).
_install_llm_accounting()

from app.routes import router
from app.langfuse_tracer import init_langfuse
from app.memory import build_all_backends
from app.agent_service import AgentService
from app.heartbeat import start as start_heartbeat

app = FastAPI(title=f"agent-{os.getenv('AGENT_ID', '?')}")
app.include_router(router)


@app.on_event("startup")
async def _startup() -> None:
    init_langfuse()
    backends = await build_all_backends()
    app.state.agent = AgentService(backends=backends)
    threading.Thread(
        target=start_http_server,
        kwargs={"port": int(os.getenv("METRICS_PORT", "9100"))},
        daemon=True,
    ).start()
    app.state.heartbeat_task = start_heartbeat()


@app.on_event("shutdown")
async def _shutdown() -> None:
    if hasattr(app.state, "heartbeat_task"):
        app.state.heartbeat_task.cancel()
    if hasattr(app.state, "agent"):
        await app.state.agent.close()


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
