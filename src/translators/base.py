from typing import List, Union

from tqdm import tqdm


class BaseTranslator:
    minibatch = 16
    text_split = 256
    max_gen_length = 384

    def _find_the_most_centred_position(self, text: str, delimiters: List[str]):
        split_positions = []
        for index, char in enumerate(text):
            if char in delimiters:
                split_positions.append(index)

        if not split_positions:
            return None

        best_split = None
        min_difference = float("inf")
        for position in split_positions:
            left_length = position + 1
            right_length = len(text) - left_length
            difference = abs(left_length - right_length)

            if difference < min_difference:
                min_difference = difference
                best_split = position

        return best_split

    def _split_text_to_chunks(self, text: str, max_len: int = 256):
        if len(text) <= max_len:
            return [text]

        def find_split_position(text: str):
            delimiters = [".", "!", "?"]
            split_pos = self._find_the_most_centred_position(text, delimiters)
            if split_pos is not None and split_pos != len(text) - 1:
                return split_pos

            delimiters = [";", ",", ":", "\n"]
            split_pos = self._find_the_most_centred_position(text, delimiters)
            if split_pos is not None and split_pos != len(text) - 1:
                return split_pos

            delimiters = ['"', "(", ")", " "]
            split_pos = self._find_the_most_centred_position(text, delimiters)
            if split_pos is not None and split_pos != len(text) - 1:
                return split_pos

            return len(text) // 2

        split_pos = find_split_position(text)

        first_chunk = text[: split_pos + 1]
        remaining_text = text[split_pos + 1 :].lstrip()
        return self._split_text_to_chunks(first_chunk, max_len) + self._split_text_to_chunks(remaining_text, max_len)

    def _batch_inference(self, texts: List[str]):
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_gen_length
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        generated_tokens = self.model.generate(
            **inputs,
            max_length=self.max_gen_length,
            num_beams=1,
            no_repeat_ngram_size=2,
        )

        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    def _init_model(self, target_lang: str):
        raise NotImplementedError

    def _simple_batching(self, texts: List[str]):
        batches = []
        indices = []
        bs = self.minibatch
        for i in range(0, len(texts), bs):
            batches.append(texts[i : i + bs])
            indices.append(list(range(i, min(i + bs, len(texts)))))
        return batches, indices

    def _process_batch(self, texts: List[str], show_progress: bool = False):
        text_mini_batches, indices = self._simple_batching(texts)

        res_texts = ["" for _ in range(len(texts))]

        results = []
        if show_progress:
            text_mini_batches = tqdm(text_mini_batches, desc="Translating texts")

        for i, text_mini_batch in enumerate(text_mini_batches):
            out = self._batch_inference(text_mini_batch)

            results.append(out)

        for ids, results_mini_batch in zip(indices, results):
            for i, result_text in enumerate(results_mini_batch):
                res_texts[ids[i]] = result_text

        return res_texts

    def __call__(self, text: Union[str, List[str]], show_progress: bool = False):
        if isinstance(text, str):
            return self.__call__([text], show_progress)[0]

        elif isinstance(text, list):
            indices = []
            text_chunks = []
            for i, chunk in enumerate(text):
                current_chunks = self._split_text_to_chunks(chunk, max_len=self.text_split)

                text_chunks += current_chunks
                indices += [i] * len(current_chunks)

            tr_text_chunks = self._process_batch(text_chunks, show_progress=show_progress)

            tr_texts = ["" for _ in range(len(text))]

            for i, tr_text in zip(indices, tr_text_chunks):
                tr_texts[i] = tr_texts[i] + " " + tr_text

            tr_texts = [text.strip() for text in tr_texts]
            return tr_texts
        else:
            raise ValueError("text must be either a string or a list of strings")

    def __init__(self, target_lang: str):
        self._init_model(target_lang)
