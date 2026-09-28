"""
Ollama chat client for the floating "Project Chat" widget
(app.orchestration.projects / app.web.project_routes).

Kept as its own client, same reasoning app.ai.trading_assistant's module
docstring already gives for being separate from app.ai.ollama_client:
each of these three clients has one deliberately narrow job --
ollama_client proposes numeric genome values, trading_assistant is
Owen's personal market-analysis assistant with his fixed strategy
prompts baked in, and this one is a general-purpose assistant scoped to
"help Owen work on this specific T58 project" with no market/strategy
content of its own. None of the three should grow to cover another's job.

Phase 1/2 scope (see app.orchestration.projects's own module docstring):
this client only chats -- it has no tool-calling / job-delegation
ability yet. It IS given a read-only summary of the project's recent
background-job activity (via app.web.job_manager.JOB_MANAGER.list_jobs)
as plain text context, so it can meaningfully discuss "how's that Search
Lab run going" even though it can't start or stop one -- delegation
(letting it actually start one) is explicitly future Phase 3 work.

Fail-safe everywhere, matching app.ai.ollama_client / trading_assistant's
own convention exactly: unreachable host, timeout, bad response -- all
of these return ("", "<human-readable reason>"), never an exception that
could take down the chat route.
"""
from __future__ import annotations

from app.ai import ollama_settings
from app.ai.ollama_settings import OllamaSettings

DEFAULT_TIMEOUT_SECONDS = 120

# Same reasoning as app.ai.trading_assistant.TradingAssistantClient's own
# identical constant: bounds how long a chatty local/CPU-only model can
# run before returning, since generation length (not connection latency)
# is the actual cause of "the assistant takes forever" on local hardware.
DEFAULT_NUM_PREDICT = 700

# How many of the project's own past turns to actually send to Ollama as
# context on each call. The full transcript is still persisted (up to
# app.orchestration.projects.MAX_STORED_MESSAGES) -- this only bounds
# the PROMPT sent on any single turn, same split app.ai.trading_assistant
# makes between "history it stores" and "history it sends".
MAX_HISTORY_TURNS_IN_PROMPT = 20


def build_system_prompt(project_name: str, activity_lines: list[str] | None = None) -> str:
    """Pure function (no network) so prompt content can be unit-tested
    without a live Ollama server -- same pattern as
    app.ai.ollama_client._build_prompt.

    activity_lines: optional pre-formatted, plain-text lines describing
    this project's recent background jobs (see
    app.web.project_routes._activity_summary_lines, itself built from
    app.web.job_manager.JOB_MANAGER.list_jobs -- deterministic data, not
    something the model computes itself). Passing this is what lets the
    assistant answer "how's my search going" meaningfully even though it
    can't start or stop jobs yet."""
    activity_section = ""
    if activity_lines:
        body = "\n".join(f"  - {line}" for line in activity_lines)
        activity_section = f"""

Recent background jobs run under this project (for your reference only --
you cannot start, stop, or modify these; just discuss them if asked):
{body}"""

    return f"""You are the project assistant for a T58 Quant Algo Backtester project called \
"{project_name}". T58 Quant Algo Backtester is Owen's own desktop/web app for backtesting \
algorithmic trading strategies against prop-firm evaluation rules and estimating pass/payout \
probability via Monte Carlo simulation (Search Lab, Evolution Lab, Full Pipeline, CPCV, \
Walk-Forward Optimization, and related tools).

Your job in this conversation is to help Owen think through and plan work on this specific \
project -- answering questions, keeping track of context across the conversation, and \
discussing the project's own background-job activity when asked. You cannot start, stop, or \
configure any backtest, search, or optimization job yourself yet, and you have no access to \
Owen's actual market data, strategy code, or backtest results beyond what's summarized below \
or what Owen tells you directly in this chat -- don't invent specific numbers, trade counts, \
or outcomes you weren't given.{activity_section}

Be direct and concise. If Owen asks you to do something you don't yet have the ability to do \
(like actually launching a Search Lab run), say so plainly rather than pretending to have done it.
"""


class ProjectChatClient:
    def __init__(self, settings: OllamaSettings, timeout: int = DEFAULT_TIMEOUT_SECONDS):
        self.settings = settings
        self.timeout = timeout

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        return headers

    def _build_messages(self, system_prompt: str, user_message: str, history: list[dict] | None) -> list[dict]:
        """history is expected in the exact {"role": ..., "content": ...}
        shape app.orchestration.projects.append_chat_message stores (a
        "ts" key, if present, is harmless -- Ollama ignores unknown
        fields) -- only the last MAX_HISTORY_TURNS_IN_PROMPT are sent,
        oldest of that window first, matching
        TradingAssistantClient._build_messages's identical trim."""
        messages = [{"role": "system", "content": system_prompt}]
        for turn in (history or [])[-MAX_HISTORY_TURNS_IN_PROMPT:]:
            messages.append({"role": turn.get("role", "user"), "content": turn.get("content", "")})
        messages.append({"role": "user", "content": user_message})
        return messages

    def chat(
        self, project_name: str, user_message: str,
        history: list[dict] | None = None, activity_lines: list[str] | None = None,
    ) -> tuple[str, str | None]:
        """Calls Ollama's /api/chat (non-streaming). Returns (reply,
        error) -- reply is "" and error is set on any failure, the exact
        fail-safe convention app.ai.ollama_client and
        app.ai.trading_assistant both already use, so callers can treat
        "AI Assist disabled" and "Ollama unreachable" identically."""
        import requests

        if not self.settings.is_usable:
            return "", "Ollama isn't enabled/configured yet. Turn it on in AI Assistant settings."

        host = (self.settings.host or "").rstrip("/")
        system_prompt = build_system_prompt(project_name, activity_lines)
        messages = self._build_messages(system_prompt, user_message, history)

        try:
            resp = requests.post(
                f"{host}/api/chat",
                headers=self._headers(),
                json={
                    "model": self.settings.model, "messages": messages, "stream": False,
                    "keep_alive": ollama_settings.INTERACTIVE_KEEP_ALIVE,
                    "options": {"num_predict": DEFAULT_NUM_PREDICT},
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            reply = (data.get("message") or {}).get("content", "")
            return reply, None
        except requests.exceptions.ConnectionError:
            return "", f"Couldn't reach Ollama at {host} (is `ollama serve` running?)."
        except requests.exceptions.Timeout:
            return "", f"Ollama at {host} didn't respond in time."
        except Exception as exc:
            return "", f"Ollama request failed: {exc}"
