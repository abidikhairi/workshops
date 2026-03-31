"""
Demo 3 — Interactive Word Vector Playground
Session 1: NLP & Word Vectors

Run: python demos/demo3_playground.py
Prereqs: pip install gensim rich prompt_toolkit
"""

import gensim.downloader as api
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

MODEL_NAME = "glove-wiki-gigaword-50"

console = Console()

COMMAND_NAMES = ["similar", "analogy", "similarity", "oddone", "help", "quit"]

HELP_LINES = [
    ("[bold cyan]similar[/]    [dim]<word>[/]", "find the 10 nearest neighbors"),
    ("[bold cyan]analogy[/]    [dim]<a> <b> <c>[/]", "compute  a - b + c  ≈  ?"),
    ("[bold cyan]similarity[/] [dim]<w1> <w2>[/]", "cosine similarity between two words"),
    ("[bold cyan]oddone[/]     [dim]<w1> <w2> <w3> ...[/]", "which word doesn't belong?"),
    ("[bold cyan]help[/]", "show this message"),
    ("[bold cyan]quit[/]", "exit"),
]


def print_help() -> None:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold")
    table.add_column(style="dim")
    for cmd, desc in HELP_LINES:
        table.add_row(cmd, desc)
    console.print(Panel(table, title="Commands", border_style="cyan", expand=False))


def score_bar(score: float, width: int = 20) -> Text:
    """Render a small horizontal bar for a similarity score."""
    filled = round(score * width)
    bar = Text()
    bar.append("█" * filled, style="green")
    bar.append("░" * (width - filled), style="dim")
    bar.append(f" {score:.3f}", style="bold")
    return bar


def check_vocab(wv, words: list[str]) -> bool:
    for w in words:
        if w not in wv:
            console.print(f"[red]'{w}' not in vocabulary[/]")
            return False
    return True


# ── Commands ─────────────────────────────────────────────────

def cmd_similar(wv, args: list[str]) -> None:
    if len(args) != 1:
        console.print("[yellow]Usage: similar <word>[/]")
        return
    word = args[0]
    if not check_vocab(wv, [word]):
        return

    table = Table(title=f"Nearest neighbors of '{word}'", border_style="blue")
    table.add_column("#", style="dim", width=3)
    table.add_column("Word", style="bold")
    table.add_column("Score")
    for i, (neighbor, score) in enumerate(wv.most_similar(word, topn=10), 1):
        table.add_row(str(i), neighbor, score_bar(score))
    console.print(table)


def cmd_analogy(wv, args: list[str]) -> None:
    if len(args) != 3:
        console.print("[yellow]Usage: analogy <a> <b> <c>   →   a - b + c ≈ ?[/]")
        return
    a, b, c = args
    if not check_vocab(wv, [a, b, c]):
        return

    console.print(
        f"\n  [bold]{a}[/] [dim]-[/] [bold]{b}[/] [dim]+[/] [bold]{c}[/] [dim]≈[/]"
    )
    table = Table(border_style="magenta", show_header=False)
    table.add_column("Word", style="bold")
    table.add_column("Score")
    for word, score in wv.most_similar(positive=[a, c], negative=[b], topn=5):
        table.add_row(word, score_bar(score))
    console.print(table)


def cmd_similarity(wv, args: list[str]) -> None:
    if len(args) != 2:
        console.print("[yellow]Usage: similarity <word1> <word2>[/]")
        return
    w1, w2 = args
    if not check_vocab(wv, [w1, w2]):
        return

    score = wv.similarity(w1, w2)
    console.print(
        f"\n  similarity([bold]{w1}[/], [bold]{w2}[/]) = ", score_bar(score)
    )


def cmd_oddone(wv, args: list[str]) -> None:
    if len(args) < 3:
        console.print("[yellow]Usage: oddone <word1> <word2> <word3> ...[/]")
        return
    if not check_vocab(wv, args):
        return

    odd = wv.doesnt_match(args)
    parts = []
    for w in args:
        if w == odd:
            parts.append(f"[bold red strikethrough]{w}[/]")
        else:
            parts.append(f"[green]{w}[/]")
    console.print(f"\n  {' · '.join(parts)}")
    console.print(f"  [dim]→[/] [bold red]{odd}[/] doesn't belong")


COMMANDS = {
    "similar": cmd_similar,
    "analogy": cmd_analogy,
    "similarity": cmd_similarity,
    "oddone": cmd_oddone,
}


# ── Main ─────────────────────────────────────────────────────

def main() -> None:
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(f"Loading {MODEL_NAME}...", total=None)
        wv = api.load(MODEL_NAME)

    console.print(
        Panel(
            f"[bold green]{len(wv):,}[/] words  ·  "
            f"[bold green]{wv.vector_size}[/]d vectors  ·  "
            f"model: [cyan]{MODEL_NAME}[/]",
            title="Word Vector Playground",
            border_style="green",
        )
    )
    print_help()

    vocab_list = list(wv.key_to_index.keys())[:50_000]
    completer = WordCompleter(COMMAND_NAMES + vocab_list, ignore_case=True)

    while True:
        try:
            line = prompt(
                HTML("<b>&gt;</b> "),
                completer=completer,
                complete_while_typing=False,
            ).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye![/]")
            break

        if not line:
            continue

        parts = line.lower().split()
        cmd, args = parts[0], parts[1:]

        if cmd in ("quit", "exit", "q"):
            console.print("[dim]Bye![/]")
            break
        elif cmd == "help":
            print_help()
        elif cmd in COMMANDS:
            COMMANDS[cmd](wv, args)
        else:
            console.print(f"[red]Unknown command: '{cmd}'. Type 'help' for usage.[/]")


if __name__ == "__main__":
    main()
