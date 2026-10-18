"""Ask a database questions in plain English, safely."""

from .agent import Agent, AgentAnswer, Step
from .database import Database, DatabaseError, QueryResult, describe_schema
from .models import (
    AnthropicModel,
    GroqModel,
    LanguageModel,
    ModelError,
    OllamaModel,
    OpenAICompatibleModel,
    ScriptedModel,
)

__version__ = "0.1.0"

__all__ = [
    "Agent", "AgentAnswer", "Step",
    "Database", "DatabaseError", "QueryResult", "describe_schema",
    "LanguageModel", "ScriptedModel", "ModelError",
    "GroqModel", "OllamaModel", "AnthropicModel", "OpenAICompatibleModel",
    "__version__",
]
