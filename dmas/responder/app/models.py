from pydantic import BaseModel

class ResponseRequest(BaseModel):
    question: str
    backend: str  # "mem0" | "graphiti" | "rag" | "cognee" | "full_context"
    # LoCoMo session date for the question's evidence ("8 May 2023 at
    # 4:42 pm"). Anchors relative time references in the answer; without
    # it the LLM falls back to its training cutoff and emits dates from
    # ~2024 instead of the conversation period.
    session_date: str = ""
    # Langfuse trace ID generated upstream (coordinator). Forwarded so
    # every responder-side openai call rolls up under the same langfuse
    # trace as the coordinator's routing call. None means the responder
    # is being called directly without a coordinator.
    trace_id: str | None = None
    # Langfuse session id (= experiment_name on the bench). Set as
    # `langfuse.session.id` on the detached `responder.respond` root
    # span so it groups under the same session as the bench's
    # `ask.question` and `load.message` traces.
    session_id: str | None = None
    # Forwarded for `langfuse.tags` (memory:<backend>, conv:<n>, mode:…)
    # so trace lists in langfuse can be filtered by experiment leg.
    conv_index: int | None = None
    mode: str | None = None