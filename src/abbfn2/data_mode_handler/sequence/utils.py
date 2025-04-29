# Copyright 2025 InstaDeep Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Protocol, Tuple

import numpy as np
import regex as re


class Tokenizer(Protocol):
    @property
    def padding_idx(self) -> int:
        pass

    @property
    def mask_idx(self) -> int:
        pass

    @property
    def normal_token_idx(self) -> list[int]:
        pass

    def tokenize(self, sequence: str) -> list[str]:
        pass

    def encode(self, sequence: str) -> np.array:
        pass


class BFNTokenizer(Tokenizer):
    """A tokenizer that extends Tokenizer with additional capabilities such
    as fixed length padding and optional cropping of sequences.

    Attributes:
        fixed_length (int): The fixed length to which all sequences will be padded or cropped.
        crop_method (str): Method of cropping if sequence length exceeds fixed_length.
            Can be 'random' for random cropping, 'error' to raise an error, or None to ignore.
        unk_token (str): Token representing unknown words.
        pad_token (str): Token used for padding shorter sequences.
        mask_token (str): Token used for masking words in sequences (e.g., in BERT).
        class_token (str): Special token often used to represent the start of a sequence in classification tasks.
        eos_token (str): Token that signifies the end of a sequence.
        bos_token (str): Token that signifies the beginning of a sequence.
        prepend_bos_token (bool): Whether to prepend the BOS token to sequences.
        prepend_cls_token (bool): Whether to prepend the CLS token to sequences.
        append_eos_token (bool): Whether to append the EOS token to sequences.
        extra_special_tokens (Optional[List[str]]): Additional special tokens that might be needed.
        tokens_to_ids (Optional[Dict[str, int]]): Mapping from tokens to their respective IDs.
    """

    def __init__(
        self,
        standard_tokens: list[str],
        fixed_length: int,
        crop_method: str = "random",
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        mask_token: str = "<mask>",
        class_token: str = "<cls>",
        eos_token: str = "<eos>",
        bos_token: str = "<bos>",
        prepend_bos_token: bool = False,
        prepend_cls_token: bool = False,
        append_eos_token: bool = False,
        extra_special_tokens: list[str] | None = None,
        tokens_to_ids: dict[str, int] | None = None,
    ):
        self._standard_tokens = standard_tokens
        self._unk_token = unk_token
        self._pad_token = pad_token
        self._mask_token = mask_token
        self._class_token = class_token
        self._eos_token = eos_token
        self._bos_token = bos_token
        self._prepend_bos_token = prepend_bos_token
        self._prepend_cls_token = prepend_cls_token
        self._append_eos_token = append_eos_token
        self._extra_special_tokens = extra_special_tokens
        self._tokens_to_ids = tokens_to_ids

        special_tokens = [
            unk_token,
            pad_token,
            mask_token,
            class_token,
            eos_token,
            bos_token,
        ]

        self._special_tokens = special_tokens
        self._all_tokens = special_tokens + standard_tokens

        self._tokens_to_ids = {tok: i for i, tok in enumerate(self._all_tokens)}
        self._ids_to_tokens = {i: tok for tok, i in self._tokens_to_ids.items()}

        self.fixed_length = fixed_length
        self.crop_method = crop_method.lower() if crop_method else "error"
        assert self.crop_method in [
            "random",
            "error",
        ], "Crop method must be 'random' or 'error'."

        self.num_special_toks = sum(
            [prepend_bos_token, prepend_cls_token, append_eos_token],
        )
        self.max_raw_length = self.fixed_length - self.num_special_toks
        self._compiled_regex = re.compile("|".join(self._all_tokens + [r"\S"]))  # noqa

    @property
    def vocabulary(self) -> List[str]:
        return self._all_tokens

    @property
    def pad_token(self) -> str:
        return self._pad_token

    @property
    def eos_token(self) -> str:
        return self._eos_token

    @property
    def padding_idx(self) -> int:
        """
        Property that returns id (int representation) of the pad token.

        Returns:
            Id (int representation) of the pad token.
        """
        return self._tokens_to_ids[self.pad_token]

    @property
    def special_tokens(self) -> List[str]:
        return self._special_tokens

    @property
    def normal_token_idx(self) -> list[int]:
        return [self._tokens_to_ids[token] for token in self._standard_tokens]

    @property
    def mask_idx(self) -> int:
        return self._tokens_to_ids[self._mask_token]

    @property
    def standard_tokens(self) -> List[str]:
        return self._standard_tokens

    def id_to_token(self, token_id: int) -> str:
        try:
            return self._ids_to_tokens[token_id]
        except KeyError:
            raise KeyError(f"Token id {token_id} not found in vocabulary")

    def _maybe_crop_sequence(self, sequence: str) -> str:
        """Conditionally crops the sequence if its length exceeds the maximum allowable raw length.

        Args:
            sequence (str): The sequence to potentially crop.

        Returns:
            str: The potentially cropped sequence.
        """
        seq_len = len(sequence)
        if seq_len > self.max_raw_length:
            if self.crop_method == "error":
                raise ValueError(
                    f"Sequence length {seq_len} exceeds the maximum length {self.max_raw_length}.",
                )
            elif self.crop_method == "random":
                start_pos = np.random.randint(0, seq_len - self.max_raw_length)
                sequence = sequence[start_pos : start_pos + self.max_raw_length]
        return sequence

    def _tokenize(self, sequence: str) -> Tuple[List[str], List[int]]:
        """
        Tokenizes a sequence and returns the list of tokens as well
        as the list of their IDs. Any character found in the sequence that does not
        correspond to any token in the vocabulary is replaced by the unk token.

        Args:
            sequence: Sequence to be tokenized.

        Returns:
            List of tokens.
            List of token ids.
        """
        tokens: List[str] = self._compiled_regex.findall(sequence)
        tokens = [
            tok if tok in self._tokens_to_ids.keys() else self._unk_token
            for tok in tokens
        ]
        if self._prepend_cls_token:
            tokens = [self._class_token] + tokens

        if self._prepend_bos_token:
            tokens = [self._bos_token] + tokens

        if self._append_eos_token:
            tokens.append(self._eos_token)

        tokens_ids = [self._tokens_to_ids[tok] for tok in tokens]

        return tokens, tokens_ids

    def tokenize(self, sequence: str) -> tuple[list[str], list[int]]:
        """Tokenizes the sequence, applying cropping if necessary, and then tokenizing via the superclass method.

        Args:
            sequence (str): The sequence to tokenize.

        Returns:
            Tuple[List[str], List[int]]: A tuple containing the list of tokens and their respective IDs.
        """
        sequence = self._maybe_crop_sequence(sequence)
        return self._tokenize(sequence)

    def batch_tokenize(self, sequences: List[str]) -> List[Tuple[List[str], List[int]]]:
        """
        Tokenizes a batch of sequences.
        Sequences are padded to the maximum length in the batch.

        Args:
            sequences: Batch of sequences to be tokenized.

        Returns:
            Batch of tokenized sequences as well as their token ids,
            where every sequence has been padded to the maximum length
            in the batch.
        """
        tmp = [self.tokenize(seq) for seq in sequences]
        ret = self.pad_tokens_batch(tmp)
        return ret

    def pad_tokens_batch(
        self,
        batch: list[tuple[list[str], list[int]]],
    ) -> list[tuple[list[str], list[int]]]:
        """Pads a batch of tokenized sequences to a fixed length.

        Args:
            batch (List[Tuple[List[str], List[int]]]): The batch of sequences with their token IDs.

        Returns:
            List[Tuple[List[str], List[int]]]: The padded list of tokenized sequences and their IDs.
        """
        lengths = [len(t[0]) for t in batch]
        deltas = [self.fixed_length - length for length in lengths]
        padded_tokens = [
            t[0] + ([self.pad_token] * delta) for t, delta in zip(batch, deltas)
        ]
        padded_tokens_ids = [
            t[1] + ([self.padding_idx] * delta) for t, delta in zip(batch, deltas)
        ]
        return [
            (toks, toks_ids) for toks, toks_ids in zip(padded_tokens, padded_tokens_ids)
        ]


class IMGTTokenizer(BFNTokenizer):
    """A Tokenizer subclass that pads sequences in an IMGT-compliant manner by inserting padding tokens in the center."""

    def pad_tokens_batch(
        self,
        batch: list[tuple[list[str], list[int]]],
    ) -> list[tuple[list[str], list[int]]]:
        """Pads a batch of tokenized sequences to a fixed length in an IMGT-compliant manner.

        Args:
            batch (List[Tuple[List[str], List[int]]]): The batch of sequences with their token IDs.

        Returns:
            List[Tuple[List[str], List[int]]]: The padded list of tokenized sequences and their IDs.
        """
        lengths = [len(t[0]) for t in batch]
        deltas = [self.fixed_length - length for length in lengths]
        mids = [(length + 1) // 2 for length in lengths]

        padded_tokens = [
            t[0][:mid] + [self.pad_token] * delta + t[0][mid:]
            for t, delta, mid in zip(batch, deltas, mids)
        ]
        padded_tokens_ids = [
            t[1][:mid] + [self.padding_idx] * delta + t[1][mid:]
            for t, delta, mid in zip(batch, deltas, mids)
        ]
        return list(zip(padded_tokens, padded_tokens_ids))
