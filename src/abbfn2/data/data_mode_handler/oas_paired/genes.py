import logging
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
from jax import Array

from abbfn2.data.data_mode_handler.base import DataModeHandler
from abbfn2.data.data_mode_handler.oas_paired.constants import (
    VALID_ALLELE_COUNTS,
    VALID_GENE_COUNTS,
)
from abbfn2.data.data_mode_handler.utils import load_from_hdf5, write_to_hdf5
from abbfn2.data.types import DataModeBatch, RawBatch


def preprocess_germline_genes(
    raw_batch: RawBatch,
    carry_args: dict,
    dm_key: str,
    unknown_id: int,
    gene2id: dict[str, int],
    include_alleles: bool = False,
) -> tuple[DataModeBatch, RawBatch, dict]:
    """Preprocesses the germline genes data.

    Args:
        raw_batch (RawBatch): The raw batch of data.
        carry_args (dict):  A dictionary to store any arguments that persist between
            data modes.
        dm_key (str): The key by which to identify the gene type to be processed.
        unknown_id (int): The index for unknown genes.
        gene2id (dict[str, int]): A dictionary mapping genes to ids for the gene type.
        include_alleles (bool): Whether to include the alleles for each gene.
            Defaults to False.

    Returns:
        Tuple[DataModeBatch, RawBatch]: The preprocessed data and the raw batch.
    """

    # Function to split the string and take the first part
    def split_gene(x):
        return x.split("*")[0] if isinstance(x, str) else x

    genes = np.array(raw_batch[dm_key])

    if not include_alleles:
        vectorised_split_gene = np.vectorize(split_gene)
        genes = vectorised_split_gene(genes)

    gene_ids = np.array([gene2id.get(gene, unknown_id) for gene in genes], dtype=int)
    gene_ids = gene_ids[..., None]  # [N, 1]

    dm_batch = DataModeBatch(
        x=gene_ids,
        mask=np.ones_like(gene_ids),
    )

    return dm_batch, raw_batch, carry_args


class GermlineGenesHandler(DataModeHandler):
    """Base class for germline genes data handlers."""

    def __init__(
        self, dm_key: str, include_alleles: bool = False, unknown_label: str = "unknown"
    ):
        """Initialise the DM handler with dicts to convert genes to classes.

        Args:
            dm_key (str): The key by which to identify the gene type to be processed.
            include_alleles (bool): Whether to include the alleles for each gene.
            Defaults to False.
            unknown_label (str): The label to be used for missing genes.
                Defaults to "unknown".
        """
        self.include_alleles = include_alleles
        self.dm_key = dm_key
        if self.include_alleles:
            self.genes_list = list(VALID_ALLELE_COUNTS[self.dm_key].keys())
            self.gene2id = {gene: idx for idx, gene in enumerate(self.genes_list)}
        else:
            self.genes_list = list(VALID_GENE_COUNTS[self.dm_key].keys())
            self.gene2id = {gene: idx for idx, gene in enumerate(self.genes_list)}

        self.id2gene = {idx: k for k, idx in self.gene2id.items()}

        # Define an "unknown" id for genes not in the valid list.
        self.unknown_id = len(self.gene2id)
        self.unknown_label = unknown_label
        self.gene2id[self.unknown_label] = self.unknown_id
        self.id2gene[self.unknown_id] = self.unknown_label

    def get_preprocess_function(
        self,
    ) -> tuple[Callable[[RawBatch], DataModeBatch], float]:
        """Defines and returns the preprocessing functions for gene call data.

        Returns:
            Tuple[Callable[[RawBatch], DataModeBatch], float]: The preprocessing
                function and the priority of the function.
        """
        preprocess_fn = partial(
            preprocess_germline_genes,
            dm_key=self.dm_key,
            unknown_id=self.unknown_id,
            gene2id=self.gene2id,
            include_alleles=self.include_alleles,
        )

        priority = 1.0
        return preprocess_fn, priority

    def sample_to_data(self, sample: Array) -> list[str]:
        """Converts an array of sample indices to a list of gene call labels.

        This method takes an array of indices and maps each index to its corresponding
        gene call label as defined in the id2gene dictionary attribute of the class.
        It supports handling of both scalar values and multidimensional arrays,
        automatically squeezing them to a 1D list if necessary. If the input sample
        is not an array or has more than one dimension after squeezing, a ValueError
        is raised.

        Args:
            sample (Array): A numpy array of sample indices. Can be a scalar, 1D, or
                multi-dimensional array.

        Raises:
            ValueError: If the input sample has more than one dimension after attempting
                to squeeze it.

        Returns:
            List[int]: A list of gene call labels corresponding to the input sample
                indices.
        """
        inp_shp = sample.shape
        if sample.ndim > 1:
            sample = sample.squeeze()

        if sample.ndim == 0:
            sample = sample[None]
        elif sample.ndim > 1:
            raise ValueError(f"Sample has unexpected shape: {inp_shp}")

        # Convert to gene call labels.
        return [self.id2gene.get(int(idx), self.unknown_label) for idx in sample]

    def data_to_sample(self, data: list[str]) -> Array:
        """Converts a list of gene call labels to a numpy array of sample indices.

        Args:
            data: A list of gene call labels.

        Returns:
            Array: A numpy array of sample indices.
        """
        return np.array([[self.gene2id.get(lab, self.unknown_id)] for lab in data])

    def save_data(
        self,
        data: np.ndarray,
        dir_path: Path,
        name: str = "properties.hdf5",
        exists_ok: bool = True,
    ) -> None:
        """Saves data to an hdf5 file."""
        path = dir_path / name
        if path.exists() and not exists_ok:
            raise FileExistsError(f"The file {path} already exists.")
        logging.info(f"Saving {self.dm_key} labels to {path}.")
        data = {self.dm_key: data}
        write_to_hdf5(path, data, delete_if_exists=not exists_ok)

    def load_data(self, dir_path: Path, name: str = "properties.hdf5") -> np.ndarray:
        """Loads data."""
        path = dir_path / name
        data = load_from_hdf5(path, keys=[self.dm_key])
        return data[self.dm_key]
