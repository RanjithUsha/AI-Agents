from dotenv import load_dotenv

load_dotenv()

from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_mistralai import ChatMistralAI
from rich import print


@tool
def get_text_length(text: str) -> int:
    """Return the number of characters in the text."""
    return len(text)


tools = [get_text_length]
tools_by_name = {t.name: t for t in tools}

llm = ChatMistralAI(model_name="ministral-8b-2512")
llm_with_tool = llm.bind_tools(tools)

messages: list = []
prompt = input("You:")
messages.append(HumanMessage(content=prompt))

result = llm_with_tool.invoke(messages)
messages.append(result)

if result.tool_calls:
    tool_call = result.tool_calls[0]
    selected_tool = tools_by_name[tool_call["name"]]
    tool_output = selected_tool.invoke(tool_call["args"])
    messages.append(
        ToolMessage(
            content=str(tool_output),
            tool_call_id=tool_call["id"],
        )
    )

final = llm_with_tool.invoke(messages)
print(final.content)
