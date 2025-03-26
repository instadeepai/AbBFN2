import logging

def create_fasta_from_sequences(l_seq: str, h_seq: str, output_file: str = "sequences.fasta") -> None:
    """
    Create a FASTA file from light and heavy chain sequences.
    
    Args:
        l_seq (str): Light chain sequence
        h_seq (str): Heavy chain sequence
        output_file (str, optional): Output FASTA file name. Defaults to "sequences.fasta".
    
    Example:
        >>> l_seq = "DIETLQSPASLAVSLGQ..."
        >>> h_seq = "EVKLQQSGPGLVTPSQS..."
        >>> create_fasta_from_sequences(l_seq, h_seq, "my_sequences.fasta")
    """
    # Remove any whitespace and validate sequences
    l_seq = l_seq.strip()
    h_seq = h_seq.strip()
    
    if not l_seq or not h_seq:
        raise ValueError("Both light and heavy chain sequences must be provided")
    
    # Create FASTA content
    fasta_content = f">L_chain\n{l_seq}\n>H_chain\n{h_seq}\n"
    
    # Write to file
    try:
        with open(output_file, 'w') as f:
            f.write(fasta_content)
    except IOError as e:
        logging.error(f"Error writing FASTA file: {e}", exc_info=True)
