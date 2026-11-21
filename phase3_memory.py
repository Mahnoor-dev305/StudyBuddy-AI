
"""
PHASE 3: MODEL + INSTRUCTIONS + MEMORY
======================================
Run interactive: python phase3_memory.py
Run auto demo:   python phase3_memory.py --demo

Memory is created by storing previous messages in a list
and sending the complete conversation history to Gemini.
"""

import os
import sys
from google import genai
from google.genai import types
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

MODEL = "gemini-flash-lite-latest"

SYSTEM_PROMPT = (
    "You are Study Buddy, a friendly tutor who explains things step-by-step "
    "and keeps answers concise. Use details the student told you earlier."
)

console = Console()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    console.print("[bold red]GEMINI_API_KEY not found![/]")
    sys.exit(1)

client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(api_version="v1beta")
)


def chat(history: list[dict]) -> str:
    """Send the conversation history to Gemini and return its reply."""

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=[
                {
                    "role": message["role"],
                    "parts": [{"text": message["content"]}]
                }
                for message in history
                if message["role"] != "system"
            ],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT
            )
        )

        return (response.text or "").strip()

    except Exception as e:
        console.print(f"[bold red]Gemini API error:[/] {e}")
        sys.exit(1)


def new_history() -> list[dict]:
    # MEMORY lives here. It stores the conversation messages.
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def send(history: list[dict], user_text: str) -> str:
    history.append({"role": "user", "content": user_text})

    reply = chat(history)

    history.append({"role": "assistant", "content": reply})

    return reply


def run_demo():
    console.rule("[bold]PHASE 3 DEMO: Memory[/]")

    history = new_history()

    for text in [
        "Hi! My name is Mahnoor and I'm studying for a biology exam.",
        "What's my name?",
        "What am I studying for?"
    ]:
        console.print(f"\n[bold cyan]You:[/] {text}")

        console.print(
            Panel(
                Markdown(send(history, text)),
                title="Study Buddy",
                border_style="green"
            )
        )

    console.print(
        f"\n[dim]History now holds {len(history)} messages.[/]"
    )


def run_interactive():
    console.rule("[bold]PHASE 3: Chat with Memory[/]")

    console.print(
        "Commands: [cyan]/history[/] show memory, "
        "[cyan]/reset[/] forget everything, "
        "[cyan]quit[/] exit\n"
    )

    history = new_history()

    while True:
        try:
            text = console.input("[bold cyan]You:[/] ").strip()

        except (EOFError, KeyboardInterrupt):
            break

        if not text:
            continue

        if text.lower() in {"quit", "exit"}:
            break

        if text == "/reset":
            history = new_history()
            console.print("[yellow]Memory wiped.[/]")
            continue

        if text == "/history":
            for message in history:
                console.print(
                    f"[dim]{message['role']:>9}:[/] "
                    f"{message['content']}"
                )
            continue

        console.print(
            Panel(
                Markdown(send(history, text)),
                title="Study Buddy",
                border_style="green"
            )
        )

    console.print("Bye!")


if __name__ == "__main__":
    run_demo() if "--demo" in sys.argv else run_interactive()
