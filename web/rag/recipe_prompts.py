"""Post-retrieval LLM prompts for recipe search (layer 3)."""

PROMPT_SHOW_MATCHING = """You are a recipe assistant.

The user searched for: "{QUERY}"

Here are matching recipes:

{RECIPES}

Instructions:
- Show all relevant recipes
- Highlight recipe names clearly
- Do not invent recipes
- Keep it concise"""


PROMPT_VAGUE = """You are a smart food search assistant.

User query: "{QUERY}"

Based on the available recipes below, find the best matches.

Rules:
- Match even if the query is incomplete
- Match similar meaning (e.g., meat → mutton, beef, veal)
- Ignore small spelling mistakes

Recipes:
{RECIPES}

Output:
- List matching recipe names
- Short description for each"""


PROMPT_EXPLAIN_MATCH = """User searched: "{QUERY}"

Explain why these recipes match the query.

Recipes:
{RECIPES}

Rules:
- Be short
- Mention keyword or similarity reason"""


PROMPT_DIRECT_RECIPE = """User wants: "{QUERY}"

From the recipes below, return the most relevant one in full format.

Rules:
- Only return one best match
- Use structured format (ingredients + steps)
- Do not add new data

Recipes:
{RECIPES}"""


def format_recipes_for_prompt(recipes: list[dict], max_chars: int = 12000) -> str:
    """Serialize retrieved records for injection into prompts."""
    parts: list[str] = []
    used = 0
    for i, r in enumerate(recipes, 1):
        title = r.get("title", "Untitled")
        ing = r.get("ingredients") or []
        inst = r.get("instructions") or []
        body = r.get("full_text") or ""
        block = (
            f"### Recipe {i}: {title}\n"
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
