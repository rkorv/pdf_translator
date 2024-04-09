from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from .base import BaseTranslator


class OpusMTTranslator(BaseTranslator):
    minibatch = 24

    def _init_model(self, target_lang: str):
        self.tokenizer = AutoTokenizer.from_pretrained(
            f"Helsinki-NLP/opus-mt-en-{target_lang}", device_map="cuda", torch_dtype=torch.float16
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            f"Helsinki-NLP/opus-mt-en-{target_lang}", device_map="cuda", torch_dtype=torch.float16
        )
