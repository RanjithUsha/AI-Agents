"""
City Intelligence — Streamlit chat UI (Human vs Bot).

Run: streamlit run streamlit_app.py
"""

from __future__ import annotations

import streamlit as st
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage

from city_tools import TOOLS, TOOLS_BY_NAME, api_key_status, build_llm

st.set_page_config(
    page_title="City Intelligence",
    page_icon="🏙️",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header {
        font-size: 1.75rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1e3a5f, #2563eb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .sub-header { color: #64748b; font-size: 0.95rem; margin-bottom: 1rem; }
    div[data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 0.5rem 0;
    }
    .tool-card {
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        background: #f8fafc;
    }
</style>
""",
    unsafe_allow_html=True,
)

MAX_LLM_STEPS = 12


def init_session() -> None:
    defaults = {
        "agent_messages": [],
        "ui_messages": [],
        "pending_tool_calls": None,
        "llm_with_tools": build_llm().bind_tools(TOOLS),
        "auto_approve_tools": False,
        "tool_decision": None,
        "run_llm_after": False,
        "tool_round": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def append_ui(role: str, content: str) -> None:
    st.session_state.ui_messages.append({"role": role, "content": content})


def pending_fingerprint() -> str:
    pending = st.session_state.pending_tool_calls or []
    parts = []
    for tc in pending:
        parts.append(f"{tc.get('name')}:{tc.get('id')}")
    return "|".join(parts)


def llm_step() -> None:
    """One model call: either final text or new pending tool batch."""
    llm = st.session_state.llm_with_tools
    messages: list[BaseMessage] = st.session_state.agent_messages

    result = llm.invoke(messages)
    messages.append(result)

    tool_calls = getattr(result, "tool_calls", None) or []
    if not tool_calls:
        text = result.content if isinstance(result.content, str) else str(result.content)
        append_ui("bot", text)
        st.session_state.tool_round = 0
        st.session_state.pending_tool_calls = None
        return

    st.session_state.pending_tool_calls = tool_calls
    st.session_state.tool_round += 1


def execute_pending_tools(approved: bool) -> None:
    tool_calls = st.session_state.pending_tool_calls or []
    messages: list[BaseMessage] = st.session_state.agent_messages

    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call.get("args") or {}
        tool_id = tool_call["id"]

        if approved:
            emoji = "🌤️" if tool_name == "get_city_weather" else "📰"
            append_ui("bot", f"{emoji} Ran **{tool_name}** with `{tool_args}`.")
            output = TOOLS_BY_NAME[tool_name].invoke(tool_args)
            messages.append(
                ToolMessage(content=str(output), tool_call_id=tool_id),
            )
        else:
            append_ui("bot", f"🚫 Denied **{tool_name}**.")
            messages.append(
                ToolMessage(
                    content="User denied this tool call.",
                    tool_call_id=tool_id,
                ),
            )

    st.session_state.pending_tool_calls = None


def process_queued_actions() -> None:
    """Run approve/deny/LLM steps at the top of each rerun (Streamlit-safe pattern)."""
    decision = st.session_state.tool_decision
    if decision in ("approve", "deny"):
        st.session_state.tool_decision = None
        approved = decision == "approve"
        with st.spinner("Running tools…" if approved else "Skipping tools…"):
            execute_pending_tools(approved=approved)
        st.session_state.run_llm_after = True

    if st.session_state.run_llm_after:
        st.session_state.run_llm_after = False
        steps = 0
        while steps < MAX_LLM_STEPS:
            steps += 1
            with st.spinner("🤖 Bot is thinking…"):
                llm_step()

            pending = st.session_state.pending_tool_calls
            if not pending:
                break

            if st.session_state.auto_approve_tools:
                execute_pending_tools(approved=True)
                continue

            break


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        keys = api_key_status()
        for name, ok in keys.items():
            st.markdown(f"{'✅' if ok else '❌'} {name} API key")

        st.session_state.auto_approve_tools = st.checkbox(
            "Auto-run tools (no approval popup)",
            value=st.session_state.auto_approve_tools,
            help="When off, you must click Approve for each batch of tool calls.",
        )

        st.markdown("---")
        st.markdown("### 🛠️ Tools")
        st.markdown("- 🌤️ **get_city_weather** — OpenWeather")
        st.markdown("- 📰 **get_city_news** — Tavily search")

        st.markdown("---")
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.agent_messages = []
            st.session_state.ui_messages = []
            st.session_state.pending_tool_calls = None
            st.session_state.tool_decision = None
            st.session_state.run_llm_after = False
            st.session_state.tool_round = 0
            st.rerun()

        st.markdown("---")
        st.caption("Approve each tool batch, or enable auto-run in the sidebar.")


def render_chat_history() -> None:
    for msg in st.session_state.ui_messages:
        role = msg["role"]
        content = msg["content"]
        if role == "human":
            with st.chat_message("user", avatar="🧑"):
                st.markdown(content)
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(content)


def render_tool_approval() -> None:
    pending = st.session_state.pending_tool_calls
    if not pending:
        return

    round_no = st.session_state.tool_round
    st.markdown(f"#### 🛠️ Tool approval required (step {round_no})")
    st.caption(
        "Weather + news may need **one or two** approvals. "
        "After you approve, tools run and the bot continues automatically."
    )

    for tc in pending:
        name = tc["name"]
        args = tc.get("args") or {}
        icon = "🌤️" if name == "get_city_weather" else "📰"
        st.markdown(
            f'<div class="tool-card">{icon} <b>{name}</b><br/>Args: <code>{args}</code></div>',
            unsafe_allow_html=True,
        )

    fp = pending_fingerprint()
    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            "✅ Approve & run",
            type="primary",
            use_container_width=True,
            key=f"approve_{fp}",
        ):
            st.session_state.tool_decision = "approve"
            st.rerun()
    with col2:
        if st.button(
            "🚫 Deny",
            use_container_width=True,
            key=f"deny_{fp}",
        ):
            st.session_state.tool_decision = "deny"
            st.rerun()


def main() -> None:
    init_session()
    process_queued_actions()
    render_sidebar()

    st.markdown('<p class="main-header">🏙️ City Intelligence</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Ask about weather and news for Indian cities — Bangalore, Hyderabad, Chennai, and more.</p>',
        unsafe_allow_html=True,
    )

    render_chat_history()
    render_tool_approval()

    awaiting = st.session_state.pending_tool_calls is not None
    prompt = st.chat_input(
        "Message the city assistant…",
        disabled=awaiting,
    )
    if awaiting:
        st.info("👆 Approve or deny the tools above before sending a new message.")

    if prompt:
        append_ui("human", prompt)
        st.session_state.agent_messages.append(HumanMessage(content=prompt))
        st.session_state.pending_tool_calls = None
        st.session_state.tool_round = 0
        st.session_state.run_llm_after = True
        st.rerun()


main()
