from typing import List

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from .base import BaseTranslator


class NLLB200Translator(BaseTranslator):

    languages_map = {
        "ru": "rus_Cyrl",
        "de": "deu_Latn",
        "fr": "fra_Latn",
        "es": "spa_Latn",
        "it": "ita_Latn",
        "pt": "por_Latn",
        "tr": "tur_Latn",
    }

    def _init_model(self, target_lang: str):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "facebook/nllb-200-distilled-600M", device_map="auto", torch_dtype=torch.float16
        )
        self.tokenizer.src_lang = "eng_Latn"

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            "facebook/nllb-200-distilled-600M", device_map="auto", torch_dtype=torch.float16
        )
        self.target_lang = self.languages_map[target_lang]

    def _batch_inference(self, texts: List[str]) -> List[str]:
        target_lang_code = self.tokenizer.lang_code_to_id[self.target_lang]

        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_gen_length
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        generated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=target_lang_code,
            max_length=self.max_gen_length,
            num_beams=1,
            no_repeat_ngram_size=2,
        )
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
