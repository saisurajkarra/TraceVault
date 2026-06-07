"""Grounded RAG answers over the REAL retrieved artifacts.

Takes the top-k real search hits as context and asks Claude for an answer that is
grounded strictly in that context and cites the real artifact ids. The Anthropic
API key is REQUIRED for synthesis: without it, ``/ask`` returns the real retrieved
context plus a clear note — it NEVER fabricates an answer, and the model is
instructed to never use outside knowledge or invent content.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

from .config import Settings
from .search import Searcher, SearchHit

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are tracevault's grounded answerer. Answer the user's question using ONLY the "
    "numbered context items provided, which are real artifacts (commits, files, and AI "
    "sessions/messages) from the user's own ingested Git and Claude Code history. "
    "Cite the artifact id in square brackets, e.g. [commit:ab12...], after each claim it "
    "supports. If the context does not contain enough information to answer, say so plainly "
    "— do NOT use outside knowledge and do NOT invent details, names, files, or events. "
    "Respond with the final answer only: no preamble, no meta-commentary about your reasoning."
)


class AskError(RuntimeError):
    """Raised when synthesis is requested but the Anthropic call fails."""


@dataclass
class AskResult:
    question: str
    answer: str | None
    used_key: bool
    note: str | None
    model: str | None
    context: list[dict[str, object]] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _format_context(hits: list[SearchHit]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        src = f"source={h.source}"
        if h.object_uri:
            src += f" object_uri={h.object_uri}"
        blocks.append(
            f"[{i}] id={h.id} kind={h.kind} {src}\n"
            f"    title: {h.title}\n"
            f"    text: {h.snippet}"
        )
    return "\n\n".join(blocks)


def ask(
    question: str,
    searcher: Searcher,
    settings: Settings,
    *,
    k: int = 6,
    repo: str | None = None,
) -> AskResult:
    """Retrieve real context, then synthesize a grounded answer if a key is present."""
    question = (question or "").strip()
    if not question:
        raise AskError("Question must not be empty.")

    hits = searcher.search(question, k=k, repo=repo)
    context = [h.as_dict() for h in hits]

    if not hits:
        return AskResult(
            question=question,
            answer=None,
            used_key=bool(settings.anthropic_api_key),
            note="No matching artifacts were found, so there is nothing to ground an answer in.",
            model=None,
            context=[],
        )

    if not settings.anthropic_api_key:
        return AskResult(
            question=question,
            answer=None,
            used_key=False,
            note=(
                "Synthesis needs an Anthropic API key (set ANTHROPIC_API_KEY). "
                "Below is the real retrieved context, with each item linked to its source artifact."
            ),
            model=None,
            context=context,
        )

    try:
        import anthropic
    except Exception as exc:  # pragma: no cover - dependency present
        raise AskError(f"The anthropic SDK is not available: {exc}") from exc

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    user_content = (
        f"Question: {question}\n\n"
        f"Context items (real ingested artifacts):\n\n{_format_context(hits)}\n\n"
        "Answer the question grounded only in these items, citing artifact ids."
    )
    try:
        message = client.messages.create(
            model=settings.anthropic_model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
    except anthropic.APIError as exc:
        raise AskError(
            f"Anthropic API call failed ({type(exc).__name__}): {exc}. "
            "No answer is fabricated; the retrieved context is still available."
        ) from exc

    answer = "".join(
        getattr(block, "text", "")
        for block in message.content
        if getattr(block, "type", None) == "text"
    ).strip()

    return AskResult(
        question=question,
        answer=answer or None,
        used_key=True,
        note=None if answer else "The model returned no text; see the retrieved context.",
        model=settings.anthropic_model,
        context=context,
    )
