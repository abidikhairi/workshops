"""
Demo 2 — GPT-2 Text Generation with Temperature
Session 2: Transformers & GPT Architecture

Run: python demos/demo2_gpt_generation.py
Prereqs: pip install transformers torch
"""

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel


def generate(prompt: str, temperature: float = 1.0, max_new_tokens: int = 40) -> str:
    """Generate text from a prompt using GPT-2."""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def demo_temperature(prompt: str) -> None:
    """Show how temperature affects generation."""
    print(f"Prompt: \"{prompt}\"\n")
    print("=" * 70)

    for temp in [0.3, 1.0, 1.5]:
        text = generate(prompt, temperature=temp)
        label = {0.3: "Low (0.3) — safe/repetitive", 1.0: "Medium (1.0) — balanced", 1.5: "High (1.5) — creative/wild"}
        print(f"\n[Temperature {label[temp]}]")
        print(text)
        print("-" * 70)


def demo_next_token_probabilities(prompt: str, top_k: int = 10) -> None:
    """Show the probability distribution for the next token."""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # Get probabilities for the next token
    logits = outputs.logits[0, -1, :]
    probs = torch.softmax(logits, dim=0)
    top_probs, top_indices = torch.topk(probs, top_k)

    print(f"\nPrompt: \"{prompt}\"")
    print(f"Top {top_k} next-token predictions:\n")
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        token = tokenizer.decode([idx])
        bar = "█" * int(prob.item() * 50) + "░" * (50 - int(prob.item() * 50))
        print(f"  {i+1:2d}. {token:15s} {bar} {prob.item():.3f}")


if __name__ == "__main__":
    # --- Temperature comparison ---
    demo_temperature("Natural language processing allows computers to")

    print("\n" + "=" * 70)

    # --- Next-token probabilities ---
    demo_next_token_probabilities("The cat sat on the")

    print()
    demo_next_token_probabilities("The president of the United States")

    # --- Try your own prompts ---
    # demo_temperature("Once upon a time in a land far away")
    # demo_next_token_probabilities("Machine learning is")
