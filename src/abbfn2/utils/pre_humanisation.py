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
from typing import Dict, List, Tuple

import abnumber
from Humatch.align import get_padded_seq
from Humatch.germline_likeness import mutate_seq_to_match_germline_likeness


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
                },
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
        input_heavy_seqs: List of heavy chain sequences
        input_light_seqs: List of light chain sequences
        lv_families: List of light chain target germline families
        hv_families: List of heavy chain target germline families
        target_score: Target germline likeness score
        allow_cdr_mutations: Whether to allow mutations in CDR regions
        fixed_imgt_positions: IMGT positions to keep fixed

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

    heavy_seqs = [
        mutate_seq_to_match_germline_likeness(
            seq, gl.lower(), target_score, allow_CDR_mutations, fixed_imgt_positions
        )
        for seq, gl in zip(heavy_seqs, h_germline_families)
    ]
    light_seqs = [
        mutate_seq_to_match_germline_likeness(
            seq, gl.lower(), target_score, allow_CDR_mutations, fixed_imgt_positions
        )
        for seq, gl in zip(light_seqs, l_germline_families)
    ]

    heavy_seqs = [seq.replace("-", "") for seq in heavy_seqs]
    light_seqs = [seq.replace("-", "") for seq in light_seqs]

    prehumanised_data = {}
    precursor_data = {}

    for name, h, l, h_input, l_input in zip(
        all_names, heavy_seqs, light_seqs, input_heavy_seqs, input_light_seqs
    ):
        prehumanised_data.update(create_sequence_data(name, h, l))
        precursor_data.update(create_sequence_data(name, h_input, l_input))

    return prehumanised_data, precursor_data
