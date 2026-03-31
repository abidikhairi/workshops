"""
Demo 3 — Interactive Transformer Playground
Session 2: Transformers & GPT Architecture

Run: python demos/demo3_playground.py
Prereqs: pip install transformers torch rich prompt_toolkit matplotlib seaborn
"""

import torch
import numpy as np
from prompt_toolkit import prompt
from prompt_toolkit.formatted_text import HTML
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

COMMAND_NAMES = ["generate", "attention", "predict", "fill", "help", "quit"]

HELP_LINES = [
    ("[bold cyan]generate[/]   [dim]<prompt>[/]", "generate text from a prompt (try different temps)"),
    ("[bold cyan]predict[/]    [dim]<prompt>[/]", "show top-10 next-token probabilities"),
    ("[bold cyan]attention[/]  [dim]<sentence>[/]", "show which words attend to which (saves heatmap)"),
    ("[bold cyan]fill[/]       [dim]<sentence with [MASK]>[/]", "fill in [MASK] using BERT"),
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
    filled = round(score * width)
    bar = Text()
    bar.append("█" * filled, style="green")
    bar.append("░" * (width - filled), style="dim")
    bar.append(f" {score:.3f}", style="bold")
    return bar


# ── Lazy model loading ──────────────────────────────────────

_gpt2_tokenizer = None
_gpt2_model = None
_bert_tokenizer = None
_bert_model = None


def get_gpt2():
    global _gpt2_tokenizer, _gpt2_model
    if _gpt2_tokenizer is None:
        from transformers import GPT2Tokenizer, GPT2LMHeadModel
        console.print("[dim]Loading GPT-2...[/]")
        _gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        _gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
        _gpt2_model.eval()
    return _gpt2_tokenizer, _gpt2_model


def get_bert():
    global _bert_tokenizer, _bert_model
    if _bert_tokenizer is None:
        from transformers import BertTokenizer, BertModel
        console.print("[dim]Loading BERT...[/]")
        _bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        _bert_model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)
        _bert_model.eval()
    return _bert_tokenizer, _bert_model


# ── Commands ────────────────────────────────────────────────

def cmd_generate(args: list[str]) -> None:
    if not args:
        console.print("[yellow]Usage: generate <prompt text>[/]")
        return

    prompt_text = " ".join(args)
    tokenizer, model = get_gpt2()
    inputs = tokenizer(prompt_text, return_tensors="pt")

    console.print(f"\n  [bold]Prompt:[/] {prompt_text}\n")
    for temp in [0.3, 1.0, 1.5]:
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=temp,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        label = {0.3: "Conservative", 1.0: "Balanced", 1.5: "Creative"}[temp]
        console.print(f"  [cyan]T={temp} ({label}):[/]")
        console.print(f"  {text}\n")


def cmd_predict(args: list[str]) -> None:
    if not args:
        console.print("[yellow]Usage: predict <prompt text>[/]")
        return

    prompt_text = " ".join(args)
    tokenizer, model = get_gpt2()
    inputs = tokenizer(prompt_text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, -1, :]
    probs = torch.softmax(logits, dim=0)
    top_probs, top_indices = torch.topk(probs, 10)

    table = Table(title=f"Next token after: \"{prompt_text}\"", border_style="blue")
    table.add_column("#", style="dim", width=3)
    table.add_column("Token", style="bold")
    table.add_column("Probability")

    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        token = tokenizer.decode([idx]).strip() or repr(tokenizer.decode([idx]))
        table.add_row(str(i + 1), token, score_bar(prob.item()))
    console.print(table)


def cmd_attention(args: list[str]) -> None:
    if not args:
        console.print("[yellow]Usage: attention <sentence>[/]")
        return

    sentence = " ".join(args)
    tokenizer, model = get_bert()
    inputs = tokenizer(sentence, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    attention = outputs.attentions[0][0, 0].numpy()

    # Print text-based attention summary for a specific token
    console.print(f"\n  [bold]Sentence:[/] {sentence}")
    console.print(f"  [bold]Tokens:[/] {tokens}\n")

    # Show attention from each content token (skip [CLS] and [SEP])
    for i, src_token in enumerate(tokens):
        if src_token in ("[CLS]", "[SEP]"):
            continue
        top_indices = np.argsort(attention[i])[::-1][:3]
        top_pairs = [(tokens[j], attention[i][j]) for j in top_indices]
        parts = ", ".join(f"{tok}({score:.2f})" for tok, score in top_pairs)
        console.print(f"  [cyan]{src_token:12s}[/] attends to → {parts}")

    # Save heatmap
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(attention, xticklabels=tokens, yticklabels=tokens, cmap="Blues", ax=ax, square=True)
        ax.set_title(f"Attention: \"{sentence}\"")
        plt.tight_layout()
        plt.savefig("attention_heatmap.png", dpi=150)
        plt.close()
        console.print(f"\n  [dim]Heatmap saved to attention_heatmap.png[/]")
    except ImportError:
        pass


def cmd_fill(args: list[str]) -> None:
    if not args:
        console.print("[yellow]Usage: fill <sentence with [MASK]>[/]")
        console.print("[yellow]Example: fill The cat [MASK] on the mat[/]")
        return

    sentence = " ".join(args)
    if "[MASK]" not in sentence.upper():
        console.print("[yellow]Sentence must contain [MASK][/]")
        return

    # Use a fill-mask pipeline
    from transformers import pipeline
    filler = pipeline("fill-mask", model="bert-base-uncased")
    results = filler(sentence.replace("[MASK]", "[MASK]").replace("[mask]", "[MASK]"))

    table = Table(title=f"Predictions for [MASK] in: \"{sentence}\"", border_style="magenta")
    table.add_column("#", style="dim", width=3)
    table.add_column("Token", style="bold")
    table.add_column("Score")

    for i, r in enumerate(results[:5], 1):
        table.add_row(str(i), r["token_str"], score_bar(r["score"]))
    console.print(table)


COMMANDS = {
    "generate": cmd_generate,
    "predict": cmd_predict,
    "attention": cmd_attention,
    "fill": cmd_fill,
}


# ── Main ────────────────────────────────────────────────────

def main() -> None:
    console.print(
        Panel(
            "[bold green]Transformer Playground[/]\n"
            "Explore attention, generation, and masked prediction interactively.",
            title="Session 2",
            border_style="green",
        )
    )
    print_help()

    while True:
        try:
            line = prompt(HTML("<b>&gt;</b> ")).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye![/]")
            break

        if not line:
            continue

        parts = line.split()
        cmd, args = parts[0].lower(), parts[1:]

        if cmd in ("quit", "exit", "q"):
            console.print("[dim]Bye![/]")
            break
        elif cmd == "help":
            print_help()
        elif cmd in COMMANDS:
            COMMANDS[cmd](args)
        else:
            console.print(f"[red]Unknown command: '{cmd}'. Type 'help' for usage.[/]")


if __name__ == "__main__":
    main()
