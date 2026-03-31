#!/usr/bin/env bash
# Demo Prep — run this before the session
set -e

pip install transformers torch matplotlib seaborn rich prompt_toolkit

# Pre-download models so demos load instantly in class
python -c "
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2LMHeadModel
print('Downloading BERT...')
BertTokenizer.from_pretrained('bert-base-uncased')
BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
print('Downloading GPT-2...')
GPT2Tokenizer.from_pretrained('gpt2')
GPT2LMHeadModel.from_pretrained('gpt2')
print('All models cached.')
"

echo ""
echo "Setup complete. Test the demos:"
echo "  python demos/demo1_attention.py"
echo "  python demos/demo2_gpt_generation.py"
