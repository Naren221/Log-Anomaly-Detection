# detector.py
from typing import Dict, List, Set, Tuple
import Levenshtein
import math

def detect_ngram(ngram_model: Dict[int, Set[Tuple[str]]], n: int, sequences: Dict[str, List[str]]) -> Dict[float, Set[str]]:
    """
    Detect anomalies using n-grams.
    Args:
        ngram_model (Dict[int, Set[Tuple[str]]]): N-gram model.
        n (int): N-gram size.
        sequences (Dict[str, List[str]]): Test sequences.
    Returns:
        Dict[float, Set[str]]: Detected anomalies for various thresholds.
    """
    detected = {}
    mn_max = 0
    for seq_id, seq in sequences.items():
        m = 0
        seq = [-1] + seq + [-1]
        for i in range(len(seq) - n + 1):
            ngram = tuple(seq[i:(i+n)])
            if ngram not in ngram_model[n]:
                m += 1
                break
        if n >= len(seq):
            mmax = 1
        else:
            mmax = n * (len(seq) - n / 2)
        mn = m / mmax
        detected[seq_id] = mn
        mn_max = max(mn_max, mn)
    for seq_id, mn in detected.items():
        if mn_max == 0:
            detected[seq_id] = 0
        else:
            detected[seq_id] = mn / mn_max
    return iterate_threshold(detected)

def iterate_threshold(dists: Dict[str, float]) -> Dict[float, Set[str]]:
    """
    Iterate over thresholds and create a dictionary of detected samples.
    Args:
        dists (Dict[str, float]): Dictionary of distances.
    Returns:
        Dict[float, Set[str]]: Detected samples for various thresholds.
    """
    detected_dict = {}
    for i in range(0, 100):
        detected = set()
        threshold = i / 100
        for seq_id, dist in dists.items():
            if dist >= threshold:
                detected.add(seq_id)
        detected_dict[threshold] = detected
    return detected_dict

def test_cluster_count_vectors(train_vectors: List[Dict[str, int]], test_vectors: Dict[str, Dict[str, int]], normal_seq_ids: Set[str], abnormal_seq_ids: Set[str], normalize: bool, idf: bool, idf_weights: Dict[str, float]) -> Dict[float, Set[str]]:
    """
    Detect anomalies using count vector clustering.
    Args:
        train_vectors (List[Dict[str, int]]): List of training count vectors.
        test_vectors (Dict[str, Dict[str, int]]): Dictionary of test count vectors.
        normal_seq_ids (Set[str]): Set of normal sequence IDs.
        abnormal_seq_ids (Set[str]): Set of abnormal sequence IDs.
        normalize (bool): Whether to normalize count vectors.
        idf (bool): Whether to use IDF weighting.
        idf_weights (Dict[str, float]): IDF weights for event types.
    Returns:
        Dict[float, Set[str]]: Detected anomalies for various thresholds.
    """
    dists = {}
    known_dists = {}
    train_vectors_set = set()
    for train_vector in train_vectors:
        train_vector_tuple = tuple(sorted(train_vector.items()))
        train_vectors_set.add(train_vector_tuple)
    train_vectors_reduced = [dict(train_vector_tuple) for train_vector_tuple in train_vectors_set]
    for seq_id, test_vector in test_vectors.items():
        test_vector_tuple = tuple(sorted(test_vector.items()))
        if test_vector_tuple in known_dists:
            dist = known_dists[test_vector_tuple]
        else:
            dist = check_count_vector(train_vectors_reduced, test_vector, normalize, idf, idf_weights)
            known_dists[test_vector_tuple] = dist
        dists[seq_id] = dist
    return iterate_threshold(dists)

def check_count_vector(train_vectors: List[Dict[str, int]], test_vector: Dict[str, int], normalize: bool, idf: bool, idf_weights: Dict[str, float]) -> float:
    """
    Compute distance of test vector to most similar train vector.
    Args:
        train_vectors (List[Dict[str, int]]): List of training count vectors.
        test_vector (Dict[str, int]): Test count vector.
        normalize (bool): Whether to normalize the count vectors.
        idf (bool): Whether to use IDF weighting.
        idf_weights (Dict[str, float]): IDF weights for event types.
    Returns:
        float: Minimum distance between test vector and training vectors.
    """
    min_dist = None
    for train_vector in train_vectors:
        manh = 0
        limit = 0
        for event_type in set(list(train_vector.keys()) + list(test_vector.keys())):
            idf_fact = 1
            if idf:
                if event_type in idf_weights:
                    idf_fact = idf_weights[event_type]
            norm_sum_train = 1
            norm_sum_test = 1
            if normalize:
                norm_sum_train = sum(train_vector.values())
                norm_sum_test = sum(test_vector.values())
            if event_type not in train_vector:
                manh += test_vector[event_type] * idf_fact / norm_sum_test
                limit += test_vector[event_type] * idf_fact / norm_sum_test
            elif event_type not in test_vector:
                manh += train_vector[event_type] * idf_fact / norm_sum_train
                limit += train_vector[event_type] * idf_fact / norm_sum_train
            else:
                manh += abs(train_vector[event_type] * idf_fact / norm_sum_train - test_vector[event_type] * idf_fact / norm_sum_test)
                limit += max(train_vector[event_type] * idf_fact / norm_sum_train, test_vector[event_type] * idf_fact / norm_sum_test)
        if min_dist is None:
            min_dist = manh / limit
        else:
            if manh / limit < min_dist:
                min_dist = manh / limit
    return min_dist

def test_edit_distance(train_sequences: Dict[str, List[str]], test_sequences: Dict[str, List[str]]) -> Dict[float, Set[str]]:
    """
    Detect anomalies using edit distance.
    Args:
        train_sequences (Dict[str, List[str]]): Training sequences.
        test_sequences (Dict[str, List[str]]): Test sequences.
    Returns:
        Dict[float, Set[str]]: Detected anomalies for various thresholds.
    """
    dists = {}
    dists_by_seq = {}
    train_sequence_set = set()
    train_sequence_unique = []
    for seq_id, train_sequence in train_sequences.items():
        train_sequence_tuple = tuple(train_sequence)
        train_sequence_set.add(train_sequence_tuple)
    train_sequence_unique = list(train_sequence_set)
    for seq_id, test_sequence in test_sequences.items():
        test_sequence_tuple = tuple(test_sequence)
        if test_sequence_tuple in dists_by_seq:
            dists[seq_id] = dists_by_seq[test_sequence_tuple]
        else:
            min_dist = 2
            for normal_event_sequence in train_sequence_unique:
                norm_fact = max(len(normal_event_sequence), len(test_sequence))
                dist = Levenshtein.distance(normal_event_sequence, test_sequence, score_cutoff=math.floor(norm_fact * min_dist)) / norm_fact
                if dist < min_dist:
                    min_dist = dist
                    best_matching_train_seq = normal_event_sequence
            dists[seq_id] = min_dist
            dists_by_seq[test_sequence_tuple] = min_dist
            train_sequence_unique.remove(best_matching_train_seq)
            train_sequence_unique = [best_matching_train_seq] + train_sequence_unique
    return iterate_threshold(dists)