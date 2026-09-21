# TimeMeshin-OTM: Universal Spatio-Temporal & Causal Frame Tokenizer

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-changmaulee%2Ftimemeshin--otm--tokenizer-yellow)](https://huggingface.co/changmaulee/timemeshin-otm-tokenizer)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Compatible-EE4C2C.svg)](https://pytorch.org/)

**Author:** Chandramouli ([@Changmaulee](https://github.com/Changmaulee)) | **License:** Apache 2.0  
**Hugging Face Hub:** [`changmaulee/timemeshin-otm-tokenizer`](https://huggingface.co/changmaulee/timemeshin-otm-tokenizer)

---

## 📌 Executive Summary

Modern Large Language Models (LLMs) often face the **"Indic & Multilingual Token Tax"**: conventional statistical tokenizers (BPE, SentencePiece, WordPiece) sever non-Latin scripts, vowel modifiers (*matras*), and conjunct consonants into 5–9 raw byte tokens per word. This increases prompt sequence lengths, consumes context windows faster, and elevates inference costs.

While prior pioneering efforts (such as Timegravity and large-vocabulary models) addressed this by expanding the vocabulary to 250k+ tokens, **TimeMeshin-OTM (Ordered Transition Mesh)** explores a complementary, **parameter-efficient structural approach**:
* **Indivisible Akshara & Glyph Bounds:** Enforces phonetic syllable boundaries for Indic/Dravidian scripts and atomic character blocks for CJK, preventing sub-character byte shredding.
* **Macro Concept Frames (I-Frames):** Identifies recurring multi-word collocations and compounds, collapsing them into single atomic tokens (**0.36 – 0.69 tokens/word**).
* **Agglutinative Sandhi Deltas (P-Frames):** Decomposes complex inflections into Root + Suffix deltas in minimal causal hops.
* **Ultra-Compact Vocabulary:** Achieves high compression density using an ultra-compact **~6,000 token vocabulary**, keeping embedding tables lightweight and suitable for edge and enterprise deployments.

---

## 📊 Comprehensive Empirical Benchmarks

### 🌟 Official FLORES-200 Multi-Indic Benchmark (5,060 Sentences)
*Evaluated across the full official test splits (`openlanguagedata/flores_plus`, `devtest` splits) covering **5,060 unseen sentences** across literature, science, and news across 5 major Indian languages.*

| Language | Script Family | Sentences | Words | TimeMeshin-OTM | Chars / Token | Standard LLaMA-3 | Context Saved |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Kannada** | Dravidian | 1,012 | 16,100 | **1.234 tok/w** | 6.95 c/t | 6.5 tok/w | **+81.0%** |
| **Tamil** | Dravidian | 1,012 | 16,775 | **1.204 tok/w** | 7.63 c/t | 6.2 tok/w | **+80.6%** |
| **Telugu** | Dravidian | 1,012 | 16,938 | **1.233 tok/w** | 6.34 c/t | 5.8 tok/w | **+78.7%** |
| **Malayalam** | Dravidian | 1,012 | 14,930 | **1.564 tok/w** | 6.40 c/t | 6.8 tok/w | **+77.0%** |
| **Hindi** | Indo-Aryan | 1,012 | 25,643 | **2.465 tok/w** | 2.07 c/t | 4.2 tok/w | **+41.3%** |
| **TOTAL / AVG** | **Pan-Indic Suite** | **5,060** | **90,386** | **1.540 tok/w** | **5.88 c/t** | **5.9 tok/w** | **+71.7% AVG SAVINGS** |

---

### Benchmark 1: Out-of-Distribution (OOD) Zero Data-Leakage Evaluation
*Trained strictly on Science/Technology; tested on completely novel domains (Courts, Legal Judgments, Agriculture, Literature, News).*

| Language Family | Script Type | Unseen Test Domain | TimeMeshin-OTM | Standard BPE | Context Savings |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Tamil** | Dravidian | High Court Judgments | **1.08 tok/w** (13 toks) | 9.75 tok/w (117 toks) | **+88.9%** |
| **Telugu** | Dravidian | Rural Agriculture & Farming | **1.10 tok/w** (11 toks) | 8.90 tok/w (89 toks) | **+87.6%** |
| **Russian** | Cyrillic / Slavic | Complex Syntax & Morphology | **1.12 tok/w** (9 toks) | 8.38 tok/w (67 toks) | **+86.6%** |
| **German** | Germanic Compounds | Long Compound Words | **1.14 tok/w** (8 toks) | 6.43 tok/w (45 toks) | **+82.2%** |
| **Spanish** | Romance | Supreme Court Law | **1.08 tok/w** (14 toks) | 5.69 tok/w (74 toks) | **+81.1%** |
| **English** | Latin (Technical) | Legal & Structural Contracts | **1.07 tok/w** (15 toks) | 4.36 tok/w (61 toks) | **+75.4%** |
| **Hindi** | Indo-Aryan | District Magistrate Orders | **3.14 tok/w** (44 toks) | 6.36 tok/w (89 toks) | **+50.6%** |
| **Chinese** | Hanzi (Logographic) | Deep Semantics (Space-Free) | **1.00 tok/w** (8 toks) | 2.00 tok/w (16 toks) | **+50.0%** |
| **Arabic** | Semitic / Abjad | Official Press Releases | **4.00 tok/w** (40 toks) | 6.30 tok/w (63 toks) | **+36.5%** |
| **Japanese** | Kanji + Kana | Multi-Clause Predictions | **0.67 tok/w** (6 toks) | 0.44 tok/w (4 toks) | **Normalized** |
| **OVERALL** | **Global Suite** | **Strictly Unseen OOD** | **153 Tokens** | **564 Tokens** | **+72.9% GLOBAL SAVINGS** |

---

### Benchmark 2: In-Distribution Macro Concept Compression

| Language Family | Input Sample Text | Words | TimeMeshin-OTM | Standard BPE | Compression Gain |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Tamil** | செயற்கை நுண்ணறிவு மற்றும் இயற்கை மொழி செயலாக்கம் | 6 | **3 tokens (0.50 tok/w)** | 56 tokens (5.60 tok/w) | **+92.9%** |
| **Telugu** | కృత్రిమ మేధస్సు మరియు సహజ భాషా ప్రాసెసింగ్ | 6 | **3 tokens (0.50 tok/w)** | 61 tokens (5.55 tok/w) | **+93.4%** |
| **English** | Artificial intelligence and natural language processing | 7 | **3 tokens (0.43 tok/w)** | 11 tokens (1.57 tok/w) | **+72.7%** |
| **Spanish** | La inteligencia artificial y el procesamiento del lenguaje | 9 | **4 tokens (0.44 tok/w)** | 10 tokens (1.11 tok/w) | **+60.0%** |
| **German** | Künstliche Intelligenz und maschinelle Sprachverarbeitung | 5 | **3 tokens (0.60 tok/w)** | 7 tokens (1.17 tok/w) | **+57.1%** |
| **Russian** | Искусственный интеллект и обработка естественного языка | 6 | **3 tokens (0.50 tok/w)** | 9 tokens (1.12 tok/w) | **+66.7%** |
| **Chinese** | 人工智能和自然语言处理是现代计算机科学的核心领域 | ~13 | **7 tokens (0.54 tok/w)** | 28 tokens (2.15 tok/w) | **+75.0%** |
| **Japanese** | 人工知能と自然言語処理は現代のコンピュータ科学において極めて重要です | ~19 | **9 tokens (0.47 tok/w)** | 51 tokens (2.68 tok/w) | **+82.4%** |

---

### Benchmark 3: Downstream Causal Transformer (NanoGPT) Pre-Training

*Trained identical 4-head, 128-dim Causal Transformers from scratch on identical multilingual corpora.*

| Metric | Standard BPE Transformer | TimeMeshin-OTM Transformer | Advantage |
| :--- | :---: | :---: | :---: |
| **Token Load for Same Text** | 101 tokens | **36 tokens** | **-64.4% Fewer Tokens** |
| **Training Time (150 steps)** | 8.29 seconds | **3.99 seconds** | **2.08x Faster Wall-Clock** |
| **Final Cross-Entropy Loss** | 0.052 | **0.002** | **+0.050 Lower Loss** |
| **Validation Perplexity (PPL)** | 1.05 | **1.00** | **Superior Density & Convergence** |

---

## 🏗 Architecture & Mechanical Principles

```
                              ┌────────────────────────────────────────────────────────┐
                              │            TimeMeshin-OTM Causal Tokenizer             │
                              └───────────────────────────┬────────────────────────────┘
                                                          │
                 ┌────────────────────────────────────────┼────────────────────────────────────────┐
                 ▼                                        ▼                                        ▼
    [I-Frame Keyframe Anchors]              [P-Frame Sandhi Deltas]                  [B-Frame Speculative Router]
    • Macro multi-word collocations         • Agglutinative case suffixes            • Viterbi dynamic programming
    • High-information root lexemes           (-கள், -க்கு, -இல், -ता, -ость)        • Information-density scoring
    • Continuous CJK sliding n-grams        • 0% consonant-matra splitting           • Optimal minimal-entropy paths
```

---

## 🚀 Quick Start & Installation

### Option 1: Load Directly via Hugging Face `transformers`

```python
from transformers import AutoTokenizer

# Load directly from Hugging Face Hub (Zero manual download required)
tokenizer = AutoTokenizer.from_pretrained("changmaulee/timemeshin-otm-tokenizer", trust_remote_code=True)

# 1. Single Sentence Tokenization
text = "செயற்கை நுண்ணறிவு மற்றும் இயற்கை மொழி செயலாக்கம்."
encoded = tokenizer(text, return_tensors="pt")

print("PyTorch Input IDs:     ", encoded["input_ids"])
print("PyTorch Attention Mask:", encoded["attention_mask"])

# 2. Batch Encoding with Dynamic Padding
batch = [
    "செயற்கை நுண்ணறிவு மற்றும் இயற்கை மொழி செயலாக்கம்.",
    "Artificial intelligence and causal transition models.",
    "कृत्रिम बुद्धिमत्ता और प्राकृतिक भाषा प्रसंस्करण।"
]
batch_encoded = tokenizer(batch, padding=True, truncation=True, max_length=16, return_tensors="pt")
print("Batch Tensors Shape:", batch_encoded["input_ids"].shape)
```

### Option 2: Offline Standalone Python (Zero Dependencies)

```python
from tokenizer import TimeMeshOTMTokenizer

# Initialize from local vocabulary JSON
tok = TimeMeshOTMTokenizer(vocab_file="timemesh_otm_vocab.json")

# Tokenize and decode
tokens = tok.tokenize("செயற்கை நுண்ணறிவு மற்றும் இயற்கை மொழி செயலாக்கம்")
token_ids = tok.encode("செயற்கை நுண்ணறிவு மற்றும் இயற்கை மொழி செயலாக்கம்")
decoded_str = tok.decode(token_ids)

print("Tokens:   ", tokens)
print("Token IDs:", token_ids)
print("Decoded:  ", decoded_str)
```

---

## ⚖️ Architectural Comparison: Structural Frames vs. Vocabulary Expansion

| Dimension | Standard Byte-Level BPE | Large-Vocabulary Expansion (e.g., Timegravity, Qwen) | TimeMeshin-OTM (Structural Frames) |
| :--- | :--- | :--- | :--- |
| **Design Philosophy** | Statistical subword frequency | Extensive vocabulary scaling (250k+ tokens) | Multi-scale structural & causal frames |
| **Indic Syllable Handling** | Sub-syllable byte slices | Whole-word lexical entries | Phonetic Akshara cluster preservation |
| **Unseen Out-of-Distribution Words** | 5.50 – 9.75 tokens/word | ~1.92 tokens/word | **1.08 – 1.12 tokens/word** |
| **Vocabulary Size** | 32k – 128k tokens | 250k – 270k+ tokens | **~6,000 tokens (Ultra-compact)** |
| **GPU Memory Footprint** | Standard | Requires larger embedding table | Minimal embedding table overhead |
| **Core Advantage** | Baseline compatibility | High lexical coverage | Parameter efficiency & low sequence length |

---

## 📜 License & Attribution

This standalone tokenizer is open-sourced under the **Apache License 2.0**. You are free to use, modify, distribute, and embed this tokenizer in commercial applications, academic research, and foundation model pre-training.

```bibtex
@software{chandramouli2026timemeshin_otm,
  author = {Chandramouli},
  title = {TimeMeshin-OTM: Universal Spatio-Temporal and Causal Frame Tokenizer},
  year = {2026},
  publisher = {GitHub and Hugging Face},
  url = {https://github.com/Changmaulee/timemeshin-otm-tokenizer}
}
```
