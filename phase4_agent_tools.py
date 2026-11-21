
"""
PHASE 4: AGENT WITH TOOLS
========================
Run interactive: python phase4_agent_tools.py
Run auto demo:   python phase4_agent_tools.py --demo

The agent can use Python tools:
1. Get current time
2. Calculator
3. Save notes
4. Read notes
"""

import ast
import operator
import re
import os
import sys
from datetime import datetime
from pathlib import Path

from google import genai
from google.genai import types
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

MODEL = "gemini-flash-lite-latest"
NOTES_FILE = Path(__file__).parent / "notes.txt"
MAX_TOOL_STEPS = 4

console = Console()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    console.print("[bold red]GEMINI_API_KEY not found![/]")
    sys.exit(1)

client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(api_version="v1beta")
)


# --------------------------------------------------
# TOOLS: Ordinary Python functions
# --------------------------------------------------

def get_time(_: str = "") -> str:
    """Return the current local date and time."""
    return datetime.now().astimezone().strftime(
        "%A, %d %B %Y, %I:%M:%S %p"
    )


# Safe calculator: only basic mathematical operations are allowed.
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(
        node.value, (int, float)
    ):
        return node.value

    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)

        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent too large")

        return _OPS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))

    raise ValueError("Unsupported expression")


def calculator(expression: str) -> str:
    """Safely calculate a mathematical expression."""
    try:
        if re.fullmatch(r"[\d\s.xX^()+\-*/%]+", expression):
            expression = (
                expression.replace("^", "**")
                .replace("x", "*")
                .replace("X", "*")
            )

        result = _eval_node(
            ast.parse(expression.strip(), mode="eval").body
        )

        return str(result)

    except ZeroDivisionError:
        return "Error: division by zero"

    except Exception as e:
        return f"Error: could not evaluate ({e})"


def save_note(text: str) -> str:
    """Save a note in notes.txt."""
    with open(NOTES_FILE, "a", encoding="utf-8") as f:
        f.write(
            f"[{datetime.now():%Y-%m-%d %H:%M}] {text}\n"
        )

    return f"Saved note: {text}"


def read_notes(_: str = "") -> str:
    """Read all saved notes."""
    if (
        not NOTES_FILE.exists()
        or not NOTES_FILE.read_text(encoding="utf-8").strip()
    ):
        return "No notes saved yet."

    return NOTES_FILE.read_text(encoding="utf-8").strip()


# Tool registry
TOOLS = {
    "get_time": get_time,
    "calculator": calculator,
    "save_note": save_note,
    "read_notes": read_notes,
}


# --------------------------------------------------
# SYSTEM INSTRUCTIONS
# --------------------------------------------------

SYSTEM_PROMPT = """
You are Study Buddy, a helpful tutor with access to Python tools.

AVAILABLE TOOLS:

- get_time() -> current date and time
- calculator(expression) -> performs mathematical calculations
- save_note(text) -> saves a note
- read_notes() -> reads saved notes

RULES:

1. If you need a tool, reply with ONLY one line:
   TOOL: tool_name(args)

2. Never guess the time or perform arithmetic yourself.
   Use the available tools.

3. After receiving a TOOL RESULT, answer the user
   in a short and friendly sentence.

4. If no tool is needed, answer normally.

5. Never invent a tool result.
"""

TOOL_PATTERN = re.compile(
    r"TOOL:\s*(\w+)\((.*)\)",
    re.IGNORECASE | re.DOTALL
)


def parse_tool_call(reply: str):
    """Find and extract a tool name and its arguments."""

    match = TOOL_PATTERN.search(reply)

    if not match:
        return None

    name = match.group(1).lower()
    args = match.group(2).strip()

    args = re.sub(r"^\w+\s*=\s*", "", args)
    args = args.strip().strip("'\"")

    return name, args


# --------------------------------------------------
# GEMINI CHAT
# --------------------------------------------------

def chat(history: list[dict]) -> str:
    """Send conversation history to Gemini."""

    try:
        contents = []

        for message in history:
            if message["role"] == "system":
                continue

            role = (
                "model"
                if message["role"] == "assistant"
                else "user"
            )

            contents.append({
                "role": role,
                "parts": [{"text": message["content"]}]
            })

        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2
            )
        )

        return (response.text or "").strip()

    except Exception as e:
        console.print(f"[bold red]Gemini API error:[/] {e}")
        sys.exit(1)


# --------------------------------------------------
# AGENT LOOP
# --------------------------------------------------

def agent_turn(history: list[dict], user_text: str) -> str:
    """One agent turn: model can request and use tools."""

    history.append({
        "role": "user",
        "content": user_text
    })

    for step in range(MAX_TOOL_STEPS):

        reply = chat(history)
        call = parse_tool_call(reply)

        if call is None:
            history.append({
                "role": "assistant",
                "content": reply
            })
            return reply

        name, args = call

        console.print(
            f"  [yellow]Model chose tool:[/] {name}({args!r})"
        )

        if name in TOOLS:
            result = TOOLS[name](args)
        else:
            result = f"Error: unknown tool '{name}'"

        console.print(
            f"  [yellow]Tool result:[/] {result}"
        )

        # Send the tool call and result back into conversation history.
        history.append({
            "role": "assistant",
            "content": reply
        })

        history.append({
            "role": "user",
            "content": (
                f"TOOL RESULT: {result}\n"
                "Now answer the user's question using this result."
            )
        })

    return "Sorry, I got stuck in a tool loop."


def new_history():
    return [{
        "role": "system",
        "content": SYSTEM_PROMPT
    }]


# --------------------------------------------------
# DEMO
# --------------------------------------------------

def run_demo():
    console.rule("[bold]PHASE 4 DEMO: Agent with Tools[/]")

    history = new_history()

    questions = [
        "What time is it?",
        "What is 123 * 456?",
        "Save a note: buy milk",
        "What notes do I have?",
        "Explain what a prime number is in one sentence."
    ]

    for text in questions:

        console.print(f"\n[bold cyan]You:[/] {text}")

        answer = agent_turn(history, text)

        console.print(
            Panel(
                Markdown(answer),
                title="Study Buddy",
                border_style="green"
            )
        )


# --------------------------------------------------
# INTERACTIVE CHAT
# --------------------------------------------------

def run_interactive():
    console.rule("[bold]PHASE 4: Study Buddy Agent[/]")

    console.print(
        "Try: what time is it, what is 17 * 23, "
        "save a note: ..., show my notes.\n"
        "Type quit to exit.\n"
    )

    history = new_history()

    while True:
        try:
            text = console.input("[bold cyan]You:[/] ").strip()

        except (EOFError, KeyboardInterrupt):
            break

        if text.lower() in {"quit", "exit"}:
            break

        if text:
            answer = agent_turn(history, text)

            console.print(
                Panel(
                    Markdown(answer),
                    title="Study Buddy",
                    border_style="green"
                )
            )

    console.print("Bye!")


if __name__ == "__main__":
    run_demo() if "--demo" in sys.argv else run_interactive()
