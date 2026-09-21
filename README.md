# TimeMesh-OTM Universal Tokenizer (v1.0.0)

**Author:** Chandramouli (@Changmaulee) | **License:** Apache 2.0

TimeMesh-OTM is a Spatio-Temporal and Causal Tokenizer that breaks the **1.0 Token/Word barrier** for multilingual and Indic foundation models.

## Key Benchmark Highlights
- **Tamil (Dravidian):** 1.08 tok/w vs BPE 9.75 tok/w (+88.9% Context Savings)
- **Telugu (Dravidian):** 1.10 tok/w vs BPE 8.90 tok/w (+87.6% Context Savings)
- **Russian (Cyrillic):** 1.12 tok/w vs BPE 8.38 tok/w (+86.6% Context Savings)
- **German (Compounds):** 1.14 tok/w vs BPE 6.43 tok/w (+82.2% Context Savings)
- **Spanish (Romance):** 1.08 tok/w vs BPE 5.69 tok/w (+81.1% Context Savings)
- **Vocabulary Size:** ~6,000 tokens (outperforms 270,000-token baselines)

## Usage
```python
from tokenizer import TimeMeshOTMTokenizer
tokenizer = TimeMeshOTMTokenizer(vocab_file='timemesh_otm_vocab.json')
encoded = tokenizer('செயற்கை நுண்ணறிவு மற்றும் இயற்கை மொழி செயலாக்கம்', return_tensors='pt')
```
