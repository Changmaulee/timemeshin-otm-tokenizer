import torch
from tokenizer import TimeMeshOTMTokenizer

tok = TimeMeshOTMTokenizer(vocab_file='timemesh_otm_vocab.json')
text = 'செயற்கை நுண்ணறிவு மற்றும் இயற்கை மொழி செயலாக்கம்.'
res = tok(text, return_tensors='pt')
print('Tokens:', tok.tokenize(text))
print('Tensor input_ids:', res['input_ids'])
print('Attention mask:', res['attention_mask'])
print('SUCCESS: Offline tokenization verified!')
