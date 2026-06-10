"""Base class shared by every LLM agent in the decision pipeline."""

import logging
from dataclasses import dataclass
from typing import Callable, Optional

from wealth_platform.llm.llm_client import LLMClient, LLMResponse

logger = logging.getLogger("wealth_platform.agents")


@dataclass
class AgentOutput:
    agent_name: str
    role: str
    report: str
    provider: str
    latency_s: float


class BaseAgent:
    """An agent is a named system prompt plus a context builder.

    Subclasses set `name`, `role`, and `system_prompt`, then call
    `self.run(context)` to produce a report. Every output is passed to the
    optional `on_output` callback so the platform can stream agent activity
    live to the dashboard.
    """

    name = "base"
    role = "Base Agent"
    system_prompt = ""
    max_tokens = 1600
    prefer_provider: Optional[str] = None

    def __init__(self, llm: LLMClient, on_output: Optional[Callable[[AgentOutput], None]] = None):
        self.llm = llm
        self.on_output = on_output

    def run(self, context: str) -> AgentOutput:
        logger.info("Agent %s running", self.name)
        response: LLMResponse = self.llm.chat(
            system=self.system_prompt,
            user=context,
            max_tokens=self.max_tokens,
            prefer=self.prefer_provider,
        )
        output = AgentOutput(
            agent_name=self.name,
            role=self.role,
            report=response.text,
            provider=response.provider,
            latency_s=response.latency_s,
        )
        if self.on_output:
            try:
                self.on_output(output)
            except Exception:  # noqa: BLE001 - streaming must never break the pipeline
                logger.exception("on_output callback failed for %s", self.name)
        return output
