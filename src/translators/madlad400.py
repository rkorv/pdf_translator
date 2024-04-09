from typing import List

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from .base import BaseTranslator


class MADLAD400Translator(BaseTranslator):

    def _init_model(self, target_lang: str):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "google/madlad400-3b-mt", device_map="auto", torch_dtype=torch.float16
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            "google/madlad400-3b-mt", device_map="auto", torch_dtype=torch.float16
        )
        self.target_lang = target_lang

    def _batch_inference(self, texts: List[str]):
        ntexts = []
        for text in texts:
            ntexts.append(f"<2{self.target_lang}> {text}")

        return super()._batch_inference(ntexts)
