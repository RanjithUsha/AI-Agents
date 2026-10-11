from langchain.tools import tool

@tool
def get_greeting(name : str) -> str:
    """Generate a greeting message for the user."""

    return f"Hello, {name}!, Welcome to the AI world"

result = get_greeting.invoke({"name": "John"})

print(result)
