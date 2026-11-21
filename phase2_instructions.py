
"""
PHASE 2: MODEL + SYSTEM INSTRUCTIONS
====================================
Run: python phase2_instructions.py

A system prompt tells the model WHO it is and HOW to behave.
The same user question gets different answers based on instructions.
"""

import os
import sys
from google import genai
from google.genai import types
from rich.console import Console
from rich.panel import Panel

MODEL = "gemini-flash-lite-latest"

console = Console()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    console.print("[bold red]GEMINI_API_KEY not found![/]")
    sys.exit(1)

client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(api_version="v1beta")
)


def ask_model(prompt: str, system: str | None = None) -> str:
    """Send a prompt with optional system instructions to Gemini."""
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system
            ) if system else None
        )
        return (response.text or "").strip()

    except Exception as e:
        console.print(f"[bold red]Gemini API error:[/] {e}")
        sys.exit(1)


if __name__ == "__main__":
    console.rule("[bold]PHASE 2: Model + Instructions[/]")

    SAME_QUESTION = "Why is the sky blue?"

    personas = {
        "No system prompt": None,
        "Pirate tutor": (
            "You are a pirate tutor. Explain things step-by-step, "
            "speak like a pirate, and keep it under 80 words."
        ),
        "Formal professor": (
            "You are a formal university professor. Use precise academic "
            "language, define terms, and keep it under 80 words."
        ),
        "ELI5 teacher": (
            "You explain things to a 5-year-old using a simple analogy. "
            "Keep it under 60 words."
        ),
    }

    console.print(
        f"\n[bold cyan]Same user prompt every time:[/] {SAME_QUESTION}"
    )

    for name, system in personas.items():
        console.print(f"\n[bold magenta]System prompt:[/] {name}")
        answer = ask_model(SAME_QUESTION, system=system)
        console.print(Panel(answer, title=name, border_style="green"))

    console.print(
        "\n[bold yellow]Takeaway:[/] Instructions shape behavior without "
        "changing the question. Still no memory, still no tools."
    )
