from langchain.tools import tool
from chef_data import RECIPES


def _clean(text: str) -> str:
    return text.strip().lower()


def _split(text: str) -> list[str]:
    return [_clean(p) for p in text.split(",") if p.strip()]


@tool
def find_recipes(ingredients: str) -> str:
    """Find recipes in the sample database that use the given ingredients.
    Input is a comma-separated list such as 'eggs, tomato, onion'.
    Returns up to 3 best matches with the ingredients still missing."""
    have = _split(ingredients)
    if not have:
        return "No ingredients given."

    scored = []
    for name, recipe in RECIPES.items():
        matched = [i for i in recipe["ingredients"] if any(h in i or i in h for h in have)]
        missing = [i for i in recipe["ingredients"] if i not in matched]
        if matched:
            scored.append((len(matched), name, missing))

    if not scored:
        return "No matching recipes found in the database."

    scored.sort(reverse=True)
    lines = []
    for count, name, missing in scored[:3]:
        extra = ", ".join(missing) if missing else "nothing"
        lines.append(f"- {name}: matches {count} ingredients; missing: {extra}")
    return "\n".join(lines)


@tool
def get_recipe_details(recipe_name: str) -> str:
    """Get the full details (ingredients, time, servings, steps) of one recipe by its exact name."""
    recipe = RECIPES.get(_clean(recipe_name))
    if recipe is None:
        return "Recipe not found. Available recipes: " + ", ".join(RECIPES)

    steps = "\n".join(f"{n}. {s}" for n, s in enumerate(recipe["steps"], start=1))
    return (
        f"{recipe_name}\n"
        f"Ingredients: {', '.join(recipe['ingredients'])}\n"
        f"Time: {recipe['time_min']} min | Servings: {recipe['servings']}\n"
        f"Steps:\n{steps}"
    )


@tool
def check_allergens(recipe_name: str, allergies: str) -> str:
    """Check whether a recipe contains any of the user's allergens.
    allergies is a comma-separated list such as 'egg, peanut'.
    Always call this before recommending a recipe to someone with allergies."""
    recipe = RECIPES.get(_clean(recipe_name))
    if recipe is None:
        return "Recipe not found, so it cannot be checked. Do not recommend it."

    user_allergies = _split(allergies)
    hits = [
        a for a in recipe["allergens"]
        if any(u in a or a in u for u in user_allergies)
    ]
    if hits:
        return f"UNSAFE: {recipe_name} contains: {', '.join(hits)}."
    return (
        f"No listed allergens in {recipe_name} match "
        f"({', '.join(user_allergies)}). This is sample data only; the user must verify ingredient labels."
    )