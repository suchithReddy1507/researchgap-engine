"""PDF -> extracted core topic/keywords, via Claude's native PDF support.

Path B from the design discussion: no local PDF-parsing library. The PDF is
sent straight to Claude as a document content block; Claude reads both the
text and the visual layout directly, which sidesteps multi-column reading-
order problems a raw text-extraction library would hit.

Separate from the main pipeline -- gaps.py and agent.py never import this,
and this never imports them.
"""
from __future__ import annotations

import base64
import json
import os


class IdeaExtractionError(Exception):
    """Message is safe to show directly to the user."""


class MissingApiKeyError(IdeaExtractionError):
    """Idea Check hard-requires ANTHROPIC_API_KEY, unlike the rest of the
    app where it's optional -- this is the one case the UI must surface
    clearly rather than letting it fail silently."""


_PROMPT = (
    "Read this document. It may be a formal academic paper, a project report, "
    "or informal notes -- read whatever is actually there.\n"
    "Identify its core research topic/idea and respond with ONLY a JSON object, "
    "no other text:\n"
    '{"title": "<best-guess title, or empty string if genuinely none>", '
    '"core_topic": "<2-4 word phrase you would type into a literature search engine '
    'to find papers specifically about this topic -- a single natural search phrase, '
    'not a list. Same style as: \\"federated learning healthcare\\" or '
    '\\"graph neural networks drug discovery\\". This is used verbatim as a search '
    'query, so keep it tight and specific to the actual subject, not the document '
    'type>", '
    '"keywords": ["<5-8 short keyword phrases capturing the core topic>"], '
    '"related_terms": ["<3-6 broader or synonymous terms/concepts for this topic '
    '-- e.g. \\"headache\\", \\"neurological disorder\\" for a migraine paper. These '
    'are NOT the primary keywords -- they exist to help match against literature that '
    'describes the same idea in different words>"], '
    '"summary": "<one or two sentence summary of the core idea>", '
    '"confidence": "<\\"high\\", \\"medium\\", or \\"low\\" -- your honest confidence '
    'that this document expresses a clear, gap-checkable research idea>"}\n'
    "If the document has no identifiable research topic or idea (e.g. it's just "
    "unrelated notes, a fragment, or illegible), still return the JSON with your "
    "best-effort core_topic/keywords/summary and set confidence to \"low\" -- do not refuse."
)


def extract_idea_from_pdf(pdf_bytes: bytes) -> dict:
    """Returns {"title", "keywords", "summary", "confidence"}. Raises
    MissingApiKeyError if no key is configured, or IdeaExtractionError for
    any other failure (bad PDF, API error, unparseable response) -- both
    carry a message meant to be shown to the user as-is."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise MissingApiKeyError("Idea Check requires ANTHROPIC_API_KEY to be configured on the engine.")

    import anthropic

    client = anthropic.Anthropic(timeout=60.0, max_retries=1)
    b64 = base64.standard_b64encode(pdf_bytes).decode("ascii")

    try:
        resp = client.messages.create(
            model=os.getenv("AGENT_MODEL", "claude-haiku-4-5-20251001"),
            max_tokens=500,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {"type": "base64", "media_type": "application/pdf", "data": b64},
                        },
                        {"type": "text", "text": _PROMPT},
                    ],
                }
            ],
        )
    except Exception as e:
        raise IdeaExtractionError(f"Could not read the PDF: {e}") from e

    try:
        text = resp.content[0].text.strip().removeprefix("```json").removesuffix("```").strip()
        data = json.loads(text)
    except (json.JSONDecodeError, IndexError, AttributeError) as e:
        raise IdeaExtractionError("Could not extract a topic from this PDF -- the response wasn't valid JSON.") from e

    keywords = [k for k in data.get("keywords", []) if isinstance(k, str) and k.strip()]
    if not keywords:
        raise IdeaExtractionError("Could not extract any keywords from this PDF -- try a different file.")

    # Supplementary only -- unlike keywords, an empty/missing list here just
    # means the landing comparison gets no extra recall, not a hard failure.
    related_terms = [t for t in data.get("related_terms", []) if isinstance(t, str) and t.strip()]

    core_topic = data.get("core_topic") or ""
    if not isinstance(core_topic, str) or not core_topic.strip():
        # Claude didn't give us a usable short phrase -- fall back to the
        # title, then to the first couple of keywords, rather than ever
        # sending an empty search query.
        core_topic = data.get("title") or " ".join(keywords[:2])

    return {
        "title": data.get("title") or "",
        "core_topic": core_topic.strip(),
        "keywords": keywords,
        "related_terms": related_terms,
        "summary": data.get("summary") or "",
        "confidence": data.get("confidence") or "low",
    }
