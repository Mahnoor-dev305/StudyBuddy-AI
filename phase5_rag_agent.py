
"""
PHASE 5: RAG - RETRIEVAL AUGMENTED GENERATION
==============================================
Run interactive: python phase5_rag_agent.py
Run auto demo:   python phase5_rag_agent.py --demo

Study Buddy retrieves information from the student's notes.
"""

import re
import os
import sys
from pathlib import Path

from google import genai
from google.genai import types
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

MODEL = "gemini-flash-lite-latest"
NOTES_PATH = Path(__file__).parent / "my_notes.txt"

CHUNK_SIZE = 500
OVERLAP = 100
TOP_K = 3
MAX_HISTORY = 6

console = Console()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    console.print("[bold red]GEMINI_API_KEY not found![/]")
    sys.exit(1)

client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(api_version="v1beta")
)

STOPWORDS = set("""
a an the is are was were be been of to in on at for and or but with as by from
this that these those it its what who whom which when where why how do does did can could should
would i me my you your we our they them their about tell explain please give
""".split())


# --------------------------------------------------
# 1. CHUNKING
# --------------------------------------------------

def chunk_text(
    text: str,
    size: int = CHUNK_SIZE,
    overlap: int = OVERLAP
) -> list[str]:
    """Split notes into small overlapping chunks."""

    text = text.strip()
    step = size - overlap

    return [
        text[i:i + size].strip()
        for i in range(0, len(text), step)
        if text[i:i + size].strip()
    ]


# --------------------------------------------------
# 2. RETRIEVAL
# --------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Convert text into searchable keywords."""

    return [
        word
        for word in re.findall(r"[a-z0-9]+", text.lower())
        if word not in STOPWORDS and len(word) > 1
    ]


def retrieve(
    question: str,
    chunks: list[str],
    k: int = TOP_K
) -> list[tuple[int, float, str]]:
    """Find the chunks containing the most question keywords."""

    question_terms = set(tokenize(question))
    scored = []

    for idx, chunk in enumerate(chunks, 1):
        words = tokenize(chunk)

        score = sum(words.count(term) for term in question_terms)

        if score > 0:
            scored.append((idx, float(score), chunk))

    scored.sort(key=lambda item: item[1], reverse=True)

    return scored[:k]


# --------------------------------------------------
# 3. AUGMENTING
# --------------------------------------------------

SYSTEM_PROMPT = """
You are Study Buddy, a tutor who answers using the student's own notes.

RULES:
1. Answer ONLY using the NOTES CONTEXT provided with each question.
2. Cite the chunk(s) you used like this: [Chunk 3].
3. If no relevant context is provided, say:
   "I couldn't find that in your notes."
4. Do not use outside knowledge to fill missing information.
5. Keep answers short and clear.
"""


def build_augmented_prompt(question: str, hits) -> str:
    """Combine retrieved notes with the user's question."""

    if not hits:
        context = "(no relevant chunks found)"
    else:
        context = "\n\n".join(
            f"[Chunk {i}]\n{text}"
            for i, _, text in hits
        )

    return (
        f"NOTES CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}"
    )


# --------------------------------------------------
# GEMINI CHAT
# --------------------------------------------------

def chat(messages: list[dict]) -> str:
    """Send conversation and retrieved context to Gemini."""

    try:
        contents = []

        for message in messages:
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
# ANSWER GENERATION
# --------------------------------------------------

def answer(
    question: str,
    chunks: list[str],
    history: list[dict]
) -> str:

    hits = retrieve(question, chunks)

    # Display retrieved chunks.
    table = Table(title="Retrieved chunks", show_lines=True)

    table.add_column("Chunk", justify="center")
    table.add_column("Score", justify="center")
    table.add_column("Preview")

    for i, score, text in hits:
        table.add_row(
            str(i),
            f"{score:.0f}",
            text[:90].replace("\n", " ") + "..."
        )

    console.print(
        table if hits else "[yellow]No matching chunks found.[/]"
    )

    # If nothing relevant was found, don't ask the model to guess.
    if not hits:
        reply = "I couldn't find that in your notes."
    else:
        messages = history[-MAX_HISTORY:]

        messages = messages + [{
            "role": "user",
            "content": build_augmented_prompt(question, hits)
        }]

        reply = chat(messages)

    # Store conversation memory.
    history.append({
        "role": "user",
        "content": question
    })

    history.append({
        "role": "assistant",
        "content": reply
    })

    return reply


# --------------------------------------------------
# LOAD NOTES
# --------------------------------------------------

def load_chunks() -> list[str]:
    """Read notes and divide them into chunks."""

    if not NOTES_PATH.exists():
        console.print(
            f"[bold red]{NOTES_PATH.name} not found.[/] "
            "Create it and add your study notes."
        )
        sys.exit(1)

    text = NOTES_PATH.read_text(encoding="utf-8")

    chunks = chunk_text(text)

    console.print(
        f"[dim]Loaded {NOTES_PATH.name}: {len(chunks)} chunks "
        f"({CHUNK_SIZE} characters, {OVERLAP} overlap)[/]"
    )

    return chunks


# --------------------------------------------------
# DEMO
# --------------------------------------------------

def run_demo():
    console.rule("[bold]PHASE 5 DEMO: RAG[/]")

    chunks = load_chunks()
    history = []

    questions = [
        "Who invented the Zorblax Protocol and when?",
        "What is the third step of the Zorblax Protocol called?",
        "What is the capital of Australia?"
    ]

    for question in questions:

        console.print(f"\n[bold cyan]You:[/] {question}")

        result = answer(question, chunks, history)

        console.print(
            Panel(
                Markdown(result),
                title="Study Buddy",
                border_style="green"
            )
        )


# --------------------------------------------------
# INTERACTIVE CHAT
# --------------------------------------------------

def run_interactive():
    console.rule("[bold]PHASE 5: Study Buddy with Knowledge[/]")

    chunks = load_chunks()
    history = []

    console.print(
        "Ask questions about your notes. "
        "Type [cyan]quit[/] to exit.\n"
    )

    while True:

        try:
            question = console.input(
                "[bold cyan]You:[/] "
            ).strip()

        except (EOFError, KeyboardInterrupt):
            break

        if question.lower() in {"quit", "exit"}:
            break

        if question:
            result = answer(question, chunks, history)

            console.print(
                Panel(
                    Markdown(result),
                    title="Study Buddy",
                    border_style="green"
                )
            )

    console.print("Bye!")


if __name__ == "__main__":
    run_demo() if "--demo" in sys.argv else run_interactive()
