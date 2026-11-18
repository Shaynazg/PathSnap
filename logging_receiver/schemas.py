from datetime import datetime, timezone
from typing import Any, Literal
from pydantic import BaseModel, Field

class LogEvent(BaseModel):
    event_id: str
    timestamp: datetime
    application_name: str = "PathSnap"
    source: Literal["frontend", "backend"]
    event_type: str
    severity: Literal["INFO", "WARNING", "ERROR", "DEBUG"]
    session_id: str | None = None
    request_id: str | None = None
    trip_id: str | None = None
    message: str
    metadata: dict[str, Any] = Field(default_factory=dict)
