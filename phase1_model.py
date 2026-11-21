
"""
PHASE 1: JUST A MODEL (Gemini)
No memory, no tools, no instructions.
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

# Gemini client configured with v1beta API
client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(api_version="v1beta")
)


def ask_model(prompt: str) -> str:
    """Send one prompt to Gemini and return its response."""
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt
        )
        return (response.text or "").strip()

    except Exception as e:
        console.print(f"[bold red]Gemini API error:[/] {e}")
        sys.exit(1)


if __name__ == "__main__":
    console.rule("[bold]PHASE 1: Just a Model[/]")

    prompts = [
        "Explain recursion in two sentences.",
        "What is my name?",
        "What time is it right now?",
    ]

    for i, prompt in enumerate(prompts, 1):
        console.print(f"\n[bold cyan]Prompt {i}:[/] {prompt}")
        answer = ask_model(prompt)
        console.print(
            Panel(answer, title="Model reply", border_style="green")
        )

    console.print(
        "\n[bold yellow]Takeaway:[/] No memory, no tools, "
        "no instructions. Just a model."
    )

