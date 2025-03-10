# preprocessor.py
import math
from typing import Dict, List, Set, Tuple

def train_cluster_count_vectors(sequences: Dict[str, List[str]], idf: bool) -> Tuple[List[Dict[str, int]], Dict[str, int], Dict[str, float]]:
    """
    Generate count vectors and IDF weights from sequences.
    Args:
        sequences (Dict[str, List[str]]): Dictionary of sequences.
        idf (bool): Whether to compute IDF weights.
    Returns:
        Tuple[List[Dict[str, int]], Dict[str, int], Dict[str, float]]: Count vectors, known event types, and IDF weights.
    """
    train_vectors = []
    known_event_types = {}
    idf_weights = {}
    cnt = 0
    for seq_id, sequence in sequences.items():
        cnt += 1
        train_vector = {}
        for part in sequence:
            if part not in known_event_types:
                known_event_types[part] = 1
            else:
                known_event_types[part] += 1
            if part in train_vector:
                train_vector[part] += 1
            else:
                train_vector[part] = 1
            if idf:
                if part in idf_weights:
                    idf_weights[part].add(seq_id)
                else:
                    idf_weights[part] = set([seq_id])
        if train_vector not in train_vectors:
            train_vectors.append(train_vector)
    N = cnt
    for event_type in idf_weights:
        idf_weights[event_type] = math.log10((1 + N) / len(idf_weights[event_type]))
    return train_vectors, known_event_types, idf_weights

def train_ngram(n: int, sequences: Dict[str, List[str]]) -> Dict[int, Set[Tuple[str]]]:
    """
    Generate n-grams from sequences.
    Args:
        n (int): N-gram size.
        sequences (Dict[str, List[str]]): Dictionary of sequences.
    Returns:
        Dict[int, Set[Tuple[str]]]: N-gram model.
    """
    ngram_model = {}
    for seq_id, seq in sequences.items():
        seq = [-1] + seq + [-1]
        if n not in ngram_model:
            ngram_model[n] = set()
        for i in range(len(seq) - n + 1):
            ngram = tuple(seq[i:(i+n)])
            ngram_model[n].add(ngram)
    return ngram_model