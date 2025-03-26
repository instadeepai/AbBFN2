import logging
from collections.abc import Callable, Iterable
from functools import partial
from pathlib import Path

import numpy as np
from Bio import Seq, SeqIO, SeqRecord
from jax import Array

from abbfn2.data.data_mode_handler.base import DataModeHandler
from abbfn2.data.data_mode_handler.sequence.utils import BFNTokenizer
from abbfn2.data.types import DataModeBatch, RawBatch

def preprocess_sequence(
    raw_batch: RawBatch,
    carry_args: dict,
    tokenizer: BFNTokenizer,
    sequence_variable: str = "sequence",
    mask_padding: bool = False,
) -> tuple[DataModeBatch, RawBatch]:
    """Tokenize the sequences data.

    Args:
        raw_batch (RawBatch): The raw batch of sequences.
        tokenizer (Tokenizer): The tokenizer to use for tokenization.
        sequence_variable (str): The name of the variable in the raw data corresponding to sequence data.

    Returns:
        DataModeBatch: The tokenized sequences and associated mask.
        RawBatch: The raw batch with the (possibly cropped) sequences.
    """
    # Extract sequence list: ['seq1..', 'seq2, ...']
    sequences: list[str] = raw_batch[sequence_variable]

    tokenized_sequences = tokenizer.batch_tokenize(sequences)
    tokens = [toks for char, toks in tokenized_sequences]

    # Remove special tokens from the sequence to obtained the cropped sequences as strings.
    special_tokens = set(tokenizer.special_tokens)
    sequences = [
        "".join(c for c in char if c not in special_tokens)
        for char, toks in tokenized_sequences
    ]

    raw_batch[sequence_variable] = sequences

    if mask_padding:
        # Create mask for padding tokens
        mask = (np.array(tokens) != tokenizer.padding_idx).astype(np.int32)
    else:
        mask = np.ones_like(tokens, dtype=np.int32)

    dm_batch = DataModeBatch(
        x=np.array(tokens),
        mask=mask,
    )

    return dm_batch, raw_batch, carry_args

class SequenceDataModeHandler(DataModeHandler):
    """Handles data modes specific to sequence data, providing necessary dataset transformations.

    Attributes:
        tokenizer (FixedSizeStandardTokenizer): The tokenizer used for preparing the data.
    """

    def __init__(
        self,
        tokenizer: BFNTokenizer,
        sequence_variable: str = "sequence",
        mask_padding: bool = True,
    ):
        """Initializes the SequenceDataModeHandler with a tokenizer.

        Args:
            tokenizer (Tokenizer): The tokenizer to use for data preparation.
            sequence_variable (str): The name of the variable in the raw data corresponding to sequence data.
        """
        self.tokenizer = tokenizer
        self.sequence_variable = sequence_variable
        self.mask_padding = mask_padding

    def get_preprocess_function(
        self,
    ) -> tuple[Callable[[RawBatch], DataModeBatch], bool, float]:
        """Defines and returns the preprocessing functions for sequence data.

        Returns:
            Tuple[Callable[[RawBatch], DataModeBatch], bool, float]: The preprocessing function, whether it is batched,
            and the priority of the function.
        """
        # Note: Preprocess_sequence is defined external to this class such that it can be pickled for
        # multiprocessing. This is necessary for efficient data loading and preprocessing.
        preprocess_fn = partial(
            preprocess_sequence,
            tokenizer=self.tokenizer,
            sequence_variable=self.sequence_variable,
            mask_padding=self.mask_padding,
        )
        priority = 0.0  # Should always be the first transformation applied.

        return preprocess_fn, priority

    def sample_to_mask(self, sample: Array) -> Array:
        """Infers a mask from a generated sample, indicating non-pad tokens with 1 and pad tokens with 0.

        Args:
            sample (Array): A sample for the sequence data.

        Returns:
            Array: An inferred mask.
        """
        return (sample != self.tokenizer.padding_idx).astype(np.int32)

    def sample_to_data(self, sample: Array) -> list[SeqRecord.SeqRecord]:
        """Converts a sample, or list of samples to a list of BioPython SeqRecord instances."""
        # Ensure sample is 2D.
        if sample.ndim == 1:
            sample = sample[None, :]
        elif sample.ndim > 2:
            raise ValueError(
                f"Sample has invalid shape {sample.shape}. Expected 1D or 2D array.",
            )

        records = []
        for index, int_sequence in enumerate(sample):
            # Convert to BioSeq instance
            seq = sample_to_string(int_sequence, self.tokenizer)

            # Create record
            records.append(
                SeqRecord.SeqRecord(
                    Seq.Seq(seq),
                    id=f"{index}",
                ),
            )

        return records
    
    def data_to_sample(self, data: Iterable[SeqRecord.SeqRecord | str]) -> Array:
        """Turns a batch of raw data into a model-friendly sample."""
        sequences = [
            str(record.seq) if isinstance(record, SeqRecord.SeqRecord) else record
            for record in data
        ]
        tokens = self.tokenizer.batch_tokenize(sequences)
        samples = np.array([toks for char, toks in tokens])
        return samples
    
    def save_data(
        self,
        data: list[SeqRecord.SeqRecord],
        dir_path: Path,
        name: str = "samples.fasta",
        exists_ok: bool = True,
    ) -> None:
        """Save the data to a FASTA file.

        This function saves a list of SeqRecord objects to a FASTA file in the specified directory
        with the given filename. If the file already exists, the function's behavior depends on the
        value of the exists_ok flag. If exists_ok is False and the file exists, a FileExistsError
        is raised. Otherwise, the existing file will be overwritten.

        Args:
            data (List[SeqRecord]): The data to save, each element a SeqRecord object.
            dir_path (Path): The directory to save the data to.
            name (str): The name of the file to save the data to. Defaults to "samples.fasta".
            exists_ok (bool): If False, raise an error if the file already exists. Defaults to True.

        Raises:
            FileExistsError: If the file already exists and exists_ok is False.
            IOError: If there is an issue writing to the file.
        """
        path = dir_path / f"{self.sequence_variable}.fasta"

        # Check if the file exists and handle according to exists_ok flag
        if path.exists() and not exists_ok:
            raise FileExistsError(f"The file {path} already exists.")

        logging.info(f"Saving sequence records to {path}.")
        # Use BioPython's SeqIO to write the SeqRecord list to a file in FASTA format
        SeqIO.write(data, path, "fasta")

    def load_data(
        self,
        dir_path: Path,
        name: str = "samples.fasta",
    ) -> SeqIO.FastaIO.FastaIterator:
        """Load data from a FASTA file.

        This function reads a FASTA file from the specified directory with the given filename and
        returns an iterator for the SeqRecord objects found in the file. This allows for efficient
        processing of sequences from large FASTA files without loading them all into memory at once.

        Args:
            dir_path (Path): The directory from which to load the FASTA file.
            name (str): The name of the FASTA file to be loaded. Defaults to "samples.fasta".

        Returns:
            SeqIO.FastaIO.FastaIterator: An iterator over the SeqRecord objects contained in the FASTA file.

        Raises:
            FileNotFoundError: If the FASTA file does not exist at the specified path.
        """
        path = dir_path / f"{self.sequence_variable}.fasta"
        if not path.exists():
            raise FileNotFoundError(f"No file found at {path}")
        return SeqIO.parse(path, "fasta")
    
def sample_to_string(sample: Array, tokenizer: BFNTokenizer) -> str:
    """'Detokenize' a sample, converting it back into string
    Args:
        sample (jnp.array): The sample of dtype integer to convert to string. Each element corresponds to an index.
        tokenizer (BaseTokenizer): The tokenizer associated with the sample.

    Returns:
        string_representation (str): The string representation of the sample.
    """
    string_representation = ""
    for idx in sample:
        idx = int(idx)
        token = tokenizer.id_to_token(idx)
        if token == tokenizer.eos_token:
            break
        if token in tokenizer.standard_tokens:
            string_representation += tokenizer.id_to_token(idx)
    return string_representation