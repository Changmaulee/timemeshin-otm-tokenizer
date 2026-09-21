import os
import re
import json
from typing import List, Optional, Dict
from transformers import PreTrainedTokenizer

class TimeMeshOTMTokenizer(PreTrainedTokenizer):
    vocab_files_names = {'vocab_file': 'timemesh_otm_vocab.json'}
    model_input_names = ['input_ids', 'attention_mask']

    def __init__(self, vocab_file: Optional[str] = 'timemesh_otm_vocab.json', unk_token='<unk>', pad_token='<pad>', bos_token='<bos>', eos_token='<eos>', model_max_length=2048, **kwargs):
        if vocab_file and os.path.exists(vocab_file):
            with open(vocab_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.vocab = data.get('vocab', {})
                self.macro_frames = set(data.get('macro_frames', []))
                self.micro_suffixes = set(data.get('micro_suffixes', []))
        else:
            self.vocab = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
            self.macro_frames = set()
            self.micro_suffixes = set()

        self.ids_to_tokens = {int(v): k for k, v in self.vocab.items()}
        super().__init__(unk_token=unk_token, pad_token=pad_token, bos_token=bos_token, eos_token=eos_token, model_max_length=model_max_length, **kwargs)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self.vocab)

    def _segment_script_units(self, text: str) -> List[str]:
        pattern = r'[\u0900-\u0D7F][\u093E-\u0D57\u0901-\u0903\u094D\u09CD\u0BCD\u0C4D]*|[\u0600-\u06FF][\u064B-\u065F\u0670]*|[\u4E00-\u9FFF]|[\u3040-\u309F]|[\u30A0-\u30FF]|[\u0400-\u04FF]+|[a-zA-Z0-9\u00C0-\u024F]+|[^\s\w]'
        return re.findall(pattern, text)

    def _tokenize(self, text: str) -> List[str]:
        units = self._segment_script_units(text)
        n = len(units)
        dp = {0: (0.0, [])}
        for i in range(n):
            if i not in dp:
                continue
            curr_cost, curr_tokens = dp[i]
            for j in range(i + 1, min(n + 1, i + 6)):
                chunk = ''.join(units[i:j])
                if chunk in self.vocab:
                    is_macro = 2.5 if chunk in self.macro_frames else 1.0
                    reward = ((len(chunk)) ** 2.2) * is_macro
                    new_cost = curr_cost - reward
                    if j not in dp or new_cost < dp[j][0]:
                        dp[j] = (new_cost, curr_tokens + [chunk])
            unit = units[i]
            j = i + 1
            new_cost = curr_cost - (len(unit) ** 1.2)
            if j not in dp or new_cost < dp[j][0]:
                dp[j] = (new_cost, curr_tokens + [unit])
        return dp.get(n, (0, units))[1]

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab.get(self.unk_token, 1))

    def _convert_id_to_token(self, index: int) -> str:
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return ''.join(tokens)
