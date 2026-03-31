"""
Demo 1 — Visualizing Self-Attention
Session 2: Transformers & GPT Architecture

Run: python demos/demo1_attention.py
Prereqs: pip install transformers torch matplotlib seaborn
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertModel


def get_attention(sentence: str, layer: int = 0, head: int = 0):
    """Extract attention weights from BERT for a given sentence."""
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)
    model.eval()

    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.attentions is a tuple of (num_layers) tensors,
    # each of shape (batch, num_heads, seq_len, seq_len)
    attention = outputs.attentions[layer][0, head].numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    return attention, tokens


def plot_attention(attention, tokens, title="Attention Weights"):
    """Plot an attention heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        attention,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="Blues",
        ax=ax,
        square=True,
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Attended to (Keys)")
    ax.set_ylabel("Attending from (Queries)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # --- The flagship example: coreference resolution ---
    sentence = "The trophy didn't fit in the suitcase because it was too big"
    print(f"Sentence: {sentence}\n")

    attention, tokens = get_attention(sentence, layer=0, head=0)
    plot_attention(attention, tokens, title="Attention — Layer 1, Head 1")

    # --- Show how "bank" attends differently in two contexts ---
    sentences = [
        "I deposited money at the bank this morning",
        "We sat by the river bank and watched the sunset",
    ]
    for sent in sentences:
        print(f"\nSentence: {sent}")
        attn, toks = get_attention(sent, layer=5, head=0)
        plot_attention(attn, toks, title=f"Attention (Layer 6, Head 1)\n\"{sent}\"")

    # --- Try your own sentence ---
    # custom = "The cat sat on the mat because it was comfortable"
    # attn, toks = get_attention(custom)
    # plot_attention(attn, toks, title=f"Attention: \"{custom}\"")
