"""CLI city agent with terminal human-in-the-loop tool approval."""

from __future__ import annotations

from typing import Any, Callable

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from rich import print

from city_tools import TOOLS, build_llm


@wrap_tool_call
def human_approval_wrap(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
) -> ToolMessage | Command[Any]:
    tool_name = (
        request.tool.name
        if request.tool
        else request.tool_call.get("name", "Unknown Tool")
    )
    tool_args = request.tool_call.get("args", {})
    tool_call_id = request.tool_call["id"]
    print(f"\n🛠️  Agent wants to run: {tool_name}")
    print(f"    Arguments: {tool_args}")
    if input("Approve? (yes/no): ").strip().lower() != "yes":
        print("🚫 Tool call denied.\n")
        return ToolMessage(
            content="User denied this tool call.",
            tool_call_id=tool_call_id,
        )
    return handler(request)


llm = build_llm()
agent = create_agent(
    llm,
    tools=TOOLS,
    system_prompt="You are a helpful city assistant.",
    middleware=[human_approval_wrap],
)


def main() -> None:
    print("City Agent | type exit to quit")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        print("Thinking... (this can take 10–30 seconds)\n")

        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
            )
            last_message = result["messages"][-1]
            print("[bold green]Assistant:[/bold green]", last_message.content, "\n")
        except Exception as exc:
            print(f"[bold red]Error:[/bold red] {exc}\n")


def _running_inside_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


if _running_inside_streamlit():
    import streamlit as st

    st.error(
        "You started the **wrong file**. `Agents.py` is for the terminal only.\n\n"
        "Stop this app (**Ctrl+C** in the terminal), then run:\n\n"
        "`streamlit run streamlit_app.py`"
    )
    st.stop()

if __name__ == "__main__":
    main()
