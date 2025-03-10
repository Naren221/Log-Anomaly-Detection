# data_loader.py
from typing import Dict, List

def load_sequences(file_path: str) -> Dict[str, List[str]]:
    """
    Load sequences from a file.
    Args:
        file_path (str): Path to the file containing sequences.
    Returns:
        Dict[str, List[str]]: A dictionary where keys are sequence IDs and values are lists of events.
    """
    sequences = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) > 1:
                sequences[parts[0]] = parts[1].split(' ')
            else:
                sequences[str(len(sequences))] = parts[0].split(' ')
    return sequences