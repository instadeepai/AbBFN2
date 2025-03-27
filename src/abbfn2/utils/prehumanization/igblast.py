import logging
import os
import re
import subprocess
from pathlib import Path

def s3_exists(
    s3_checkpoint_dir: str,
    endpoint: str = "https://storage.googleapis.com",
) -> bool:
    """Checks if a file or folder exists in an S3 bucket.

    Args:
        s3_checkpoint_dir (str): The S3 directory path to check for existence.
        endpoint (str, optional): The S3 endpoint. Defaults to "https://storage.googleapis.com".

    Returns:
        bool: True if the folder exists, False otherwise.
    """
    cmd = f"aws s3 ls {s3_checkpoint_dir} --endpoint={endpoint} && echo RES:1 || echo RES:0"

    # Run the command without terminal output
    result = subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    # Use regex to search for the pattern 'RES:' followed by an integer
    match = re.search(r"RES:(\d)", result.stdout)
    if match:
        return bool(int(match.group(1)))

    # If the pattern wasn't found, return False
    return False


def s3_cp(
    src: str,
    trg: str,
    endpoint: str = "https://storage.googleapis.com",
) -> subprocess.Popen:
    """Wrapper for aws s3 cp.

    Args:
        src_dir (str): Source path.
        trg_dir (str): Target path.
        endpoint (str, optional): The S3 endpoint. Defaults to "https://storage.googleapis.com".

    Returns:
        subprocess.Popen: Subprocess object that can be used to wait for the process to complete.
    """
    cmd = f"aws s3 cp {src.resolve()} {trg.resolve()} --endpoint={endpoint}"

    return subprocess.Popen(
        cmd,
        shell=True,
    )


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
    igblast_path: str | Path | None = "../../../igblast/ncbi-igblast-1.22.0/bin/igblastp",
    v_gene_db_path: str | Path | None = "../../../igblast/human/human_imgt_v_db",
    j_gene_db_path: str | Path | None = "../../../igblast/human/human_imgt_j_db",
    local_igblast_raw: str | Path | None = "../../../igblast/igblast_output_raw.tsv",
) -> dict:
    """Run the IgBLAST pipeline on antibody sequences.

    Args:
        input_file (str | Path): Path to input FASTA file containing antibody sequences
        species (str, optional): Species for gene regions - one of "human", "rat", "mouse", "rhesus_monkey". Defaults to "human".
        n_alignments (int, optional): Number of alignments to generate. Defaults to 5.
        igblast_path (str | Path | None, optional): Path to IgBLAST executable. If None, fetches from S3. Defaults to None.
        v_gene_db_path (str | Path | None, optional): Path to V gene database. If None, fetches from S3. Defaults to None.
        j_gene_db_path (str | Path | None, optional): Path to J gene database. If None, fetches from S3. Defaults to None.

    Returns:
        dict: Dictionary containing parsed IgBLAST results with alignment summaries and hit tables
    """
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

    v_genes = [data[ab]["top_v_hit"][0].split('-')[0] for ab in data]

    return v_genes