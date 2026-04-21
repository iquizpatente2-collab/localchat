"""Post-retrieval LLM prompts for manual record search (layer 3)."""

PROMPT_SHOW_MATCHING = """You are an industrial manual assistant.

The user searched for: "{QUERY}"

Here are matching manual records:

{RECIPES}

Instructions:
- Show all relevant records
- Highlight procedure/section titles clearly
- Do not invent fault codes, values, or procedures
- Keep it concise"""


PROMPT_VAGUE = """You are a smart technical-manual search assistant.

User query: "{QUERY}"

Based on the available manual records below, find the best matches.

Rules:
- Match even if the query is incomplete
- Match similar meaning (e.g., alarm/fault/warning variants, OCR typos)
- Ignore small spelling mistakes

Manual records:
{RECIPES}

Output:
- List matching record/procedure titles
- Short reason for each"""


PROMPT_EXPLAIN_MATCH = """User searched: "{QUERY}"

Explain why these manual records match the query.

Manual records:
{RECIPES}

Rules:
- Be short
- Mention keyword or similarity reason"""


PROMPT_DIRECT_RECIPE = """User wants: "{QUERY}"

From the manual records below, return the most relevant one in full format.

Rules:
- Only return one best match
- Use structured format (record title + parameters/components + steps)
- Do not add new data

Manual records:
{RECIPES}"""


def format_recipes_for_prompt(recipes: list[dict], max_chars: int = 12000) -> str:
    """Serialize retrieved manual records for prompt injection."""
    parts: list[str] = []
    used = 0
    for i, r in enumerate(recipes, 1):
        title = r.get("title", "Untitled")
        ing = r.get("ingredients") or []
        inst = r.get("instructions") or []
        body = r.get("full_text") or ""
        block = (
            f"### Record {i}: {title}\n"
            f"Page: {r.get('page', '?')}\n"
            f"Ingredients:\n"
            + "\n".join(f"  - {x}" for x in ing)
            + "\nInstructions:\n"
            + "\n".join(f"  {j}. {x}" for j, x in enumerate(inst, 1))
            + f"\nFull text:\n{body}\n"
        )
        if used + len(block) > max_chars:
            block = block[: max(0, max_chars - used)] + "\n[... truncated ...]\n"
        parts.append(block)
        used += len(block)
        if used >= max_chars:
            break
    return "\n".join(parts) if parts else "(no recipes)"
