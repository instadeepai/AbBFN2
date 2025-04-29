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

import logging
import os
import platform
import subprocess
from pathlib import Path


def run_igblast(
    fasta_file: str | Path,
    igblast_path: str | Path,
    out_path: str | Path,
    v_gene_db: str | Path,
    j_gene_db: str | Path,
    species: str | None = "human",
    n_alignments: int = 1,
):
    """Runs IgBLAST given an input fasta file.

    Args:
        fasta_file (str | Path): Input file of antibody sequences
        igblast_path (str | Path): Path to the IgBLAST executable
        out_path (str | Path): Output file path
        v_gene_db (str | Path): Path to the V gene database to be used.
        j_gene_db (str | Path): Path to the J gene database to be used.
        species (str): The species to use to find gene regions. Defaults to "human".
        n_alignments (int): _description_. Defaults to 1.
    """
    subprocess.run(
        [
            igblast_path,
            "-query",
            fasta_file,
            "-germline_db_V",
            v_gene_db,
            "-db",
            j_gene_db,
            "-organism",
            species,
            "-outfmt",
            "7",
            "-num_descriptions",
            str(n_alignments),
            "-num_alignments",
            str(n_alignments),
            "-out",
            out_path,
        ],
        check=True,
        stderr=True,
    )


def convert_strings_to_floats(d):
    """Converts strings representing floats in a dictionary to actual floats.

    Args:
      d: The input dictionary.

    Returns:
      A new dictionary with float values where applicable.
    """
    new_dict = {}
    for key, value in d.items():
        try:
            new_dict[key] = float(value)
        except ValueError:
            new_dict[key] = value
    return new_dict


def get_top_hit(hit_table: list, chain_type="V"):
    """Returns the best hit for a gene type in parsed IgBLAST output.

    Also returns the percent identity and E-value.

    Args:
        hit_table (_type_): _description_
        chain_type (str, optional): _description_. Defaults to "V".

    Returns:
        _type_: _description_
    """
    gl_hits = [hit for hit in hit_table if hit["chain_type"] == chain_type]
    return (
        gl_hits[0]["subject_id"],
        gl_hits[0]["percent_identity"],
        gl_hits[0]["evalue"],
    )


def parse_igblastp_output(file_path: str | Path):
    """Parses IgBLAST output.

    IGBLAST output in format "-outfmt 7" is a commented tsv, with comments denoted by
    "#". For each query, two tables are generated: the per-region alignment scores
    and the top hits for the given database entries (along with metadata on the
    alignment).

    Args:
        file_path (str | Path): Path to IgBLAST output file.

    Returns:
        list[dict]: The parsed IgBLAST output.
    """
    with open(file_path) as file:
        lines = file.readlines()

    queries = []
    current_query = {}
    alignment_summary = {}
    hit_table = []
    section = None

    for line in lines:
        line = line.strip()
        if line.startswith("# Query:"):
            # Save previous query data if exists
            if current_query:
                current_query["alignment_summary"] = alignment_summary
                current_query["hit_table"] = hit_table
                current_query["top_v_hit"] = get_top_hit(hit_table, chain_type="V")
                current_query["top_j_hit"] = get_top_hit(hit_table, chain_type="N/A")
                queries.append(current_query)
                current_query = {}
                alignment_summary = {}
                hit_table = []
            # Start new query
            current_query["query_id"] = line.split(":", 1)[1].strip()
        elif line.startswith("# Alignment summary"):
            section = "alignment_summary"
            # Skip the header line(s)
            continue
        elif line.startswith("# Hit table"):
            section = "hit_table"
            # Skip the header lines (next two lines)
            continue
        elif line.startswith("#") or line == "":
            # We might still be in a section
            continue
        else:
            if section == "alignment_summary":
                # Parse alignment summary line
                tokens = line.split("\t")
                region = tokens[0]
                region_dat = {
                    field: (float(tokens[i]) if tokens[i] != "N/A" else None)
                    for i, field in enumerate(
                        [
                            "from",
                            "to",
                            "length",
                            "matches",
                            "mismatches",
                            "gaps",
                            "percent_identity",
                        ],
                        start=1,
                    )
                }
                alignment_summary[region] = region_dat
            elif section == "hit_table":
                tokens = line.split()
                hits = {
                    field: tokens[i]
                    for i, field in enumerate(
                        [
                            "chain_type",
                            "query_id",
                            "subject_id",
                            "percent_identity",
                            "alignment_length",
                            "mismatches",
                            "gap_opens",
                            "gaps",
                            "q_start",
                            "q_end",
                            "s_start",
                            "s_end",
                            "evalue",
                            "bit_score",
                        ]
                    )
                }
                hits = convert_strings_to_floats(hits)
                hit_table.append(hits)

    # Append the last query
    if current_query:
        current_query["alignment_summary"] = alignment_summary
        current_query["hit_table"] = hit_table
        current_query["top_v_hit"] = get_top_hit(hit_table, chain_type="V")
        current_query["top_j_hit"] = get_top_hit(hit_table, chain_type="N/A")
        queries.append(current_query)

    data = {
        q["query_id"]: {
            "alignment_summary": q["alignment_summary"],
            "hit_table": q["hit_table"],
            "top_v_hit": q["top_v_hit"],
            "top_j_hit": q["top_j_hit"],
        }
        for q in queries
    }

    return data


def run_igblast_pipeline(
    input_file: str | Path,
    species: str = "human",
    n_alignments: int = 1,
    igblast_path: str | Path | None = None,
    v_gene_db_path: str | Path | None = "../../../igblast/human_db/human_imgt_v_db",
    j_gene_db_path: str | Path | None = "../../../igblast/human_db/human_imgt_j_db",
    local_igblast_raw: str | Path | None = "../../../igblast/igblast_output_raw.tsv",
) -> dict:
    """Run the IgBLAST pipeline on antibody sequences.

    Args:
        input_file (str | Path): Path to input FASTA file containing antibody sequences
        species (str, optional): Species for gene regions. Defaults to "human".
        n_alignments (int, optional): Number of alignments to generate. Defaults to 1.
        igblast_path (str | Path | None, optional): Path to IgBLAST executable. If None, uses appropriate version for platform.
        v_gene_db_path (str | Path | None, optional): Path to V gene database.
        j_gene_db_path (str | Path | None, optional): Path to J gene database.
        local_igblast_raw (str | Path | None, optional): Path to local IgBLAST output file.

    Returns:
        dict: Dictionary containing parsed IgBLAST results with alignment summaries and hit tables
    """
    if igblast_path is None:
        # Use Mac-specific version on macOS, Linux version otherwise
        if platform.system() == "Darwin":
            igblast_path = "../../../igblast/ncbi-igblast-1.22.0-mac/bin/igblastp"
        else:
            igblast_path = "../../../igblast/ncbi-igblast-1.22.0-linux/bin/igblastp"

    local_input_file = Path(input_file).absolute()
    v_gene_db_path = Path(v_gene_db_path).absolute()
    j_gene_db_path = Path(j_gene_db_path).absolute()
    igblast_path = Path(igblast_path).absolute()

    local_igblast_raw = Path(local_igblast_raw).absolute()
    Path(local_igblast_raw).parent.mkdir(parents=True, exist_ok=True)

    os.chdir(igblast_path.parent.parent)
    logging.info("Running IgBLAST...")
    run_igblast(
        local_input_file,
        igblast_path,
        local_igblast_raw,
        v_gene_db_path,
        j_gene_db_path,
        species=species,
        n_alignments=n_alignments,
    )
    logging.info("Parsing and cleaning IgBLAST output...")
    data = parse_igblastp_output(local_igblast_raw)
    os.remove(local_igblast_raw)
    os.chdir("../..")

    v_genes = [data[ab]["top_v_hit"][0].split("-")[0] for ab in data]

    return v_genes
