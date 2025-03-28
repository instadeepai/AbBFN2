import logging
from typing import Dict, List, Tuple

from Bio import SeqIO
import abnumber

from Humatch.align import get_padded_seq
from Humatch.germline_likeness import mutate_seq_to_match_germline_likeness


def load_precursor_sequences(fasta_path: str) -> Tuple[Dict[str, str], List[str]]:
    """
    Load precursor sequences from a FASTA file.
    
    Args:
        fasta_path: Path to the FASTA file containing precursor sequences
        
    Returns:
        Tuple containing:
            - Dictionary of sequence IDs to sequences
            - List of unique antibody names (without _H/_L suffix)
    """
    precursors = {}
    try:
        for record in SeqIO.parse(fasta_path, "fasta"):
            precursors[record.id] = str(record.seq)
    except Exception as e:
        logging.error(f"Failed to load FASTA file {fasta_path}: {e}")
        raise

    all_names = [k.split("_")[0] for k in precursors.keys() if k.endswith("_H")]
    return precursors, all_names


def load_igblast_data(heavy_path: str, light_path: str) -> Tuple[List[str], List[str]]:
    """
    Load IgBlast data for heavy and light chains.
    
    Args:
        heavy_path: Path to heavy chain IgBlast results
        light_path: Path to light chain IgBlast results
        
    Returns:
        Tuple of heavy and light chain germline families
    """
    try:
        with open(heavy_path) as rp:
            lines = [l for l in rp.readlines() if l.startswith("V\t")]
        h_germline_families = [l.split()[2].split("-")[0][2:] for l in lines]

        with open(light_path) as rp:
            lines = [l for l in rp.readlines() if l.startswith("V\t")]
        l_germline_families = [l.split()[2].split("-")[0][2:] for l in lines]
        
        return h_germline_families, l_germline_families
    except Exception as e:
        logging.error(f"Failed to load IgBlast data: {e}")
        raise


def process_sequences(
    sequences: List[str],
    germline_families: List[str],
    target_score: float,
    allow_cdr_mutations: bool,
    fixed_imgt_positions: List[int]
) -> List[str]:
    """
    Process sequences with germline mutations.
    
    Args:
        sequences: List of sequences to process
        germline_families: List of corresponding germline families
        target_score: Target germline likeness score
        allow_cdr_mutations: Whether to allow mutations in CDR regions
        fixed_imgt_positions: IMGT positions to keep fixed
        
    Returns:
        List of processed sequences
    """
    padded_seqs = [get_padded_seq(seq) for seq in sequences]
    edited_seqs = [
        mutate_seq_to_match_germline_likeness(
            seq, gl, target_score, allow_cdr_mutations, fixed_imgt_positions
        )
        for seq, gl in zip(padded_seqs, germline_families)
    ]
    return [seq.replace("-", "") for seq in edited_seqs]


def create_sequence_data(name: str, heavy_seq: str, light_seq: str) -> Dict:
    """
    Create structured data for a sequence pair using ANARCI/abnumber.
    
    Args:
        name: Antibody name
        heavy_seq: Heavy chain sequence
        light_seq: Light chain sequence
        
    Returns:
        Dictionary containing structured sequence data
    """
    try:
        h_numbered = abnumber.Chain(heavy_seq, scheme="imgt", cdr_definition="imgt")
        l_numbered = abnumber.Chain(light_seq, scheme="imgt", cdr_definition="imgt")
        
        return {
            name: {
                "h": {
                    "seq": heavy_seq,
                    "cdr1": h_numbered.cdr1_seq,
                    "cdr2": h_numbered.cdr2_seq,
                    "cdr3": h_numbered.cdr3_seq,
                    "fwr1": h_numbered.fr1_seq,
                    "fwr2": h_numbered.fr2_seq,
                    "fwr3": h_numbered.fr3_seq,
                    "fwr4": h_numbered.fr4_seq,
                },
                "l": {
                    "seq": light_seq,
                    "cdr1": l_numbered.cdr1_seq,
                    "cdr2": l_numbered.cdr2_seq,
                    "cdr3": l_numbered.cdr3_seq,
                    "fwr1": l_numbered.fr1_seq,
                    "fwr2": l_numbered.fr2_seq,
                    "fwr3": l_numbered.fr3_seq,
                    "fwr4": l_numbered.fr4_seq,
                }
            }
        }
    except Exception as e:
        logging.error(f"Failed to process sequences for {name}: {e}")
        raise


def process_prehumanisation(
    input_heavy_seqs: List[str],
    input_light_seqs: List[str],
    lv_families: List[str],
    hv_families: List[str],
    target_score: float = 0.40,
    allow_CDR_mutations: bool = False,
    fixed_imgt_positions: List[int] = [],
) -> Tuple[Dict, Dict, List[str], List[str]]:
    """
    Process pre-humanisation sequences and return results.
    
    Args:
        fasta_path: Path to input FASTA file with precursor sequences
        heavy_igblast_path: Path to heavy chain IgBlast results
        light_igblast_path: Path to light chain IgBlast results
        target_score: Target germline likeness score
        allow_cdr_mutations: Whether to allow mutations in CDR regions
        fixed_imgt_positions: IMGT positions to keep fixed
        log_level: Logging level
        
    Returns:
        Tuple containing:
            - Dictionary of prehumanised sequence data
            - Dictionary of precursor sequence data
            - List of heavy chain germline targets
            - List of light chain germline targets
    """  
    all_names = [f"Chain_{i}" for i in range(len(input_heavy_seqs))]

    heavy_seqs = [get_padded_seq(h) for h in input_heavy_seqs]
    light_seqs = [get_padded_seq(l) for l in input_light_seqs]

    # [2:] for each hv_family
    h_germline_families = [hv_families[i][2:] for i in range(len(heavy_seqs))]
    l_germline_families = [lv_families[i][2:] for i in range(len(light_seqs))]

    print(h_germline_families)
    print(l_germline_families)
    print(heavy_seqs)
    print(light_seqs)

    heavy_seqs = [mutate_seq_to_match_germline_likeness(seq, gl.lower(), target_score, allow_CDR_mutations, fixed_imgt_positions) for seq, gl in zip(heavy_seqs, h_germline_families)]
    light_seqs = [mutate_seq_to_match_germline_likeness(seq, gl.lower(), target_score, allow_CDR_mutations, fixed_imgt_positions) for seq, gl in zip(light_seqs, l_germline_families)]

    print("After mutation")
    print(heavy_seqs)
    print(light_seqs)

    heavy_seqs = [seq.replace("-", "") for seq in heavy_seqs]
    light_seqs = [seq.replace("-", "") for seq in light_seqs]

    logging.info("Creating output data")
    prehumanised_data = {}
    precursor_data = {}
    
    for name, h, l, h_input, l_input in zip(all_names, heavy_seqs, light_seqs, input_heavy_seqs, input_light_seqs):
        prehumanised_data.update(create_sequence_data(name, h, l))
        precursor_data.update(create_sequence_data(name, h_input, l_input))
    
    logging.info("Processing completed successfully")
    return prehumanised_data, precursor_data