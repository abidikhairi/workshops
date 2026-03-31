#!/usr/bin/env bash
# Demo Prep — run this before the session
set -e

pip install nltk gensim rich prompt_toolkit
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')"

echo ""
echo "Setup complete. Test the demos:"
echo "  python demos/demo1_tokenization.py"
echo "  python demos/demo2_word2vec.py"
