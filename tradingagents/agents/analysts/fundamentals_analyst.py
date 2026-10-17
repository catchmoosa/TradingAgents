from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.analysts.fundamentals_check import footnote, unmatched_figures
from tradingagents.agents.analysts.turn import WRAP_UP, take_turn
from tradingagents.agents.context import get_instrument_context_from_state, get_language_instruction
from tradingagents.agents.tools import (
    get_balance_sheet,
    get_cashflow,
    get_earnings_context,
    get_fundamentals,
    get_fundamentals_packet,
    get_income_statement,
    get_insider_transactions,
)
from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.fundamentals_packet import packet_summary_markdown

PACKET_NAME = "get_fundamentals_packet"

RAW_TOOLS = (
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement,
    get_earnings_context,
    get_insider_transactions,
)


def tools_for_run():
    """Packet by default; raw statement tools only when ``fundamentals_raw_tools`` is on."""
    tools = (get_fundamentals_packet,)
    if get_config().get("fundamentals_raw_tools"):
        tools += RAW_TOOLS
    return tools


# Import-time default (packet only). Graph setup calls ``tools_for_run``.
TOOLS = (get_fundamentals_packet,)


def _packet_from_messages(messages) -> str | None:
    for message in messages:
        name = getattr(message, "name", None)
        if name == PACKET_NAME and getattr(message, "content", None):
            return str(message.content)
        if isinstance(message, ToolMessage) and (message.name or "") == PACKET_NAME:
            return str(message.content)
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.startswith("[get_fundamentals_packet returned]"):
            return content.split("\n", 1)[-1]
    return None


def _call_packet(state) -> str:
    from tradingagents.dataflows.fundamentals_packet import render_fundamentals_packet

    return render_fundamentals_packet(state["company_of_interest"], state["trade_date"])


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)
        tools = tools_for_run()

        system_message = (
            "You are a researcher tasked with analyzing fundamental information about a company "
            "from the FUNDAMENTALS PACKET. Call `get_fundamentals_packet` (dated as of today) "
            "and write a comprehensive report from it. All figures come from the packet: quote "
            "them exactly with their period label. Do not compute new sums, ratios or TTM figures; "
            "if a figure you need is absent, say it is not in the packet. Use the packet's last "
            "close as the current price; never use moving averages as the price. Use the fiscal "
            "labels exactly as the packet writes them. Read the Data-quality flags section; when "
            "a CONFLICT is listed, use the packet's chosen value. Yahoo \"reported\" EPS is street/"
            "adjusted, labelled beside GAAP/IFRS EPS. Runway leads with TTM FCF; subsequent raises "
            "and credit lines are separate lines, not mixed into period figures. When a pro forma "
            "runway is shown it is derived (D) from as-reported cash plus subsequent equity inflows. "
            "Append a Markdown table at the end with a `Packet field` column so claims are traceable."
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " Report what your tools support; another agent decides the trade."
                    " You have access to the following tools: {tool_names}."
                    " Today's date is {current_date}; treat it as 'now' for all analysis and tool-call date ranges. {instrument_context}\n"
                    "{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        messages = list(state["messages"])
        result, report = take_turn(prompt, llm, tools, messages)

        packet_md = _packet_from_messages([*messages, result])
        if report and not packet_md:
            try:
                packet_md = _call_packet(state)
            except Exception as exc:  # noqa: BLE001 — never block the run
                packet_md = f"(fundamentals packet unavailable: {exc})"
            forced = HumanMessage(content=f"[get_fundamentals_packet returned]\n{packet_md}")
            result, report = take_turn(prompt, llm, tools, [*messages, forced])
            messages = [*messages, forced]

        if report and packet_md and get_config().get("fundamentals_number_check", True):
            missing = unmatched_figures(report, packet_md)
            if missing:
                retry = HumanMessage(
                    content=(
                        "These figures are not in the packet: "
                        + ", ".join(missing[:20])
                        + "; correct or remove them. Quote packet figures with their period labels. "
                        "Do not compute new sums."
                    )
                )
                result, report = take_turn(
                    prompt, llm, tools, [*messages, result, retry, HumanMessage(WRAP_UP)],
                )
                still = unmatched_figures(report, packet_md) if report else missing
                if still:
                    report = (report or "") + footnote(still)

        out = {
            "messages": [result],
            "fundamentals_report": report,
        }
        if packet_md:
            out["fundamentals_packet_summary"] = packet_summary_markdown(packet_md)
        return out

    return fundamentals_analyst_node
