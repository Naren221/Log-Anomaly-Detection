# main.py
import argparse
from data_loader import load_sequences
from preprocessor import train_cluster_count_vectors, train_ngram
from detector import detect_ngram, test_cluster_count_vectors, test_edit_distance
from evaluator import evaluate, print_results

def evaluate_all(data_dir: str, time_det: bool, normalize: bool, time_window: float = None):
    # Load training and test sequences
    train_sequences = load_sequences(f"{data_dir}/{data_dir.split('_')[0]}_train")
    test_normal_sequences = load_sequences(f"{data_dir}/{data_dir.split('_')[0]}_test_normal")
    test_abnormal_sequences = load_sequences(f"{data_dir}/{data_dir.split('_')[0]}_test_abnormal")

    # Merge test sequences
    test_sequences = {**test_normal_sequences, **test_abnormal_sequences}

    results = []

    # Detection based on new events
    known_event_types = set()
    for seq in train_sequences.values():
        known_event_types.update(seq)
    detected_new_events = {seq_id for seq_id, seq in test_sequences.items() if any(event not in known_event_types for event in seq)}
    tp, fn, tn, fp = evaluate(detected_new_events, test_normal_sequences.keys(), test_abnormal_sequences.keys())
    results.append(print_results('New event detection', tp, fn, tn, fp, None, 0))

    # Detection based on sequence lengths
    min_length = min(len(seq) for seq in train_sequences.values())
    max_length = max(len(seq) for seq in train_sequences.values())
    detected_length = {seq_id for seq_id, seq in test_sequences.items() if len(seq) < min_length or len(seq) > max_length}
    tp, fn, tn, fp = evaluate(detected_length, test_normal_sequences.keys(), test_abnormal_sequences.keys())
    results.append(print_results('Sequence length detection', tp, fn, tn, fp, None, 0))

    # Count vector clustering
    train_vectors, _, _ = train_cluster_count_vectors(train_sequences, idf=False)
    test_vectors = {seq_id: {event: seq.count(event) for event in seq} for seq_id, seq in test_sequences.items()}
    detected_dict = test_cluster_count_vectors(train_vectors, test_vectors, test_normal_sequences.keys(), test_abnormal_sequences.keys(), normalize, idf=False, idf_weights={})
    best_threshold = max(detected_dict.keys())
    tp, fn, tn, fp = evaluate(detected_dict[best_threshold], test_normal_sequences.keys(), test_abnormal_sequences.keys())
    results.append(print_results('Count vector clustering', tp, fn, tn, fp, best_threshold, 0))

    # 2-gram detection
    ngram_model = train_ngram(2, train_sequences)
    detected_ngram = detect_ngram(ngram_model, 2, test_sequences)
    best_threshold = max(detected_ngram.keys())
    tp, fn, tn, fp = evaluate(detected_ngram[best_threshold], test_normal_sequences.keys(), test_abnormal_sequences.keys())
    results.append(print_results('2-gram detection', tp, fn, tn, fp, best_threshold, 0))

    # Edit distance detection
    detected_edit = test_edit_distance(train_sequences, test_sequences)
    best_threshold = max(detected_edit.keys())
    tp, fn, tn, fp = evaluate(detected_edit[best_threshold], test_normal_sequences.keys(), test_abnormal_sequences.keys())
    results.append(print_results('Edit distance detection', tp, fn, tn, fp, best_threshold, 0))

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--time_det", type=bool, default=False, help="Enable time-based detection")
    parser.add_argument("--normalize", type=bool, default=True, help="Normalize count vectors")
    parser.add_argument("--time_window", type=float, default=None, help="Time window for time-based detection")
    args = parser.parse_args()

    results = evaluate_all(args.data_dir, args.time_det, args.normalize, args.time_window)
    for result in results:
        print(result)