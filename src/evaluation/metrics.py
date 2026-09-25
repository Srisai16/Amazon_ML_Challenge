"""
Evaluation Metrics Module for Amazon ML Challenge 2026.
Calculates Macro F_beta score (beta=0.5) exactly as defined in problem criteria.
"""

from typing import Dict, List, Set, Tuple


def calculate_entity_f_beta(
    true_matches: Set[str],
    pred_matches: Set[str],
    beta: float = 0.5
) -> float:
    """
    Computes F_beta score for a single Source 1 entity.
    
    Rules:
    - If true_matches is empty (singleton):
        - If pred_matches is empty -> 1.0 (correct singleton detection)
        - If pred_matches is NOT empty -> 0.0 (false merge on singleton)
    - If true_matches is not empty:
        - If pred_matches is empty -> 0.0
        - Otherwise compute standard precision, recall, and F_beta:
          F_0.5 = (1.25 * P * R) / (0.25 * P + R)
    """
    # Singleton case
    if len(true_matches) == 0:
        return 1.0 if len(pred_matches) == 0 else 0.0

    # Non-singleton but predicted empty
    if len(pred_matches) == 0:
        return 0.0

    tp = len(true_matches.intersection(pred_matches))
    if tp == 0:
        return 0.0

    precision = tp / len(pred_matches)
    recall = tp / len(true_matches)

    beta_sq = beta ** 2  # 0.25 for beta=0.5
    numerator = (1.0 + beta_sq) * (precision * recall)
    denominator = (beta_sq * precision) + recall

    if denominator == 0:
        return 0.0

    return numerator / denominator


def compute_macro_f05(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]]
) -> Tuple[float, Dict[str, float]]:
    """
    Computes macro-averaged F0.5 across all Source 1 entities in ground truth.
    
    Args:
        ground_truth: Dict mapping source1_entity_id -> set of true matched entity_ids.
        predictions: Dict mapping source1_entity_id -> set of predicted entity_ids.
        
    Returns:
        macro_f05: Overall mean score across all entities.
        per_entity_scores: Dict mapping entity_id -> score.
    """
    scores = {}
    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        score = calculate_entity_f_beta(true_set, pred_set, beta=0.5)
        scores[s1_id] = score

    macro_f05 = sum(scores.values()) / len(scores) if scores else 0.0
    return macro_f05, scores


def compute_blocking_metrics(
    ground_truth: Dict[str, Set[str]],
    candidates: Dict[str, Set[str]],
    total_target_pool_size: int
) -> Dict[str, float]:
    """
    Evaluates candidate generation / blocking quality:
    - Pair Completeness (Recall ceiling)
    - Reduction Ratio
    - Average Candidate Count per S1
    """
    total_true_pairs = sum(len(s) for s in ground_truth.values())
    recalled_pairs = 0
    total_cand_pairs = 0

    for s1_id, true_set in ground_truth.items():
        cand_set = candidates.get(s1_id, set())
        recalled_pairs += len(true_set.intersection(cand_set))
        total_cand_pairs += len(cand_set)

    num_s1 = len(ground_truth)
    pair_completeness = recalled_pairs / total_true_pairs if total_true_pairs > 0 else 1.0
    avg_candidates = total_cand_pairs / num_s1 if num_s1 > 0 else 0.0

    max_possible_comparisons = num_s1 * total_target_pool_size
    reduction_ratio = 1.0 - (total_cand_pairs / max_possible_comparisons) if max_possible_comparisons > 0 else 1.0

    return {
        "pair_completeness_recall": pair_completeness,
        "reduction_ratio": reduction_ratio,
        "avg_candidates_per_s1": avg_candidates,
        "total_candidate_pairs": total_cand_pairs,
        "total_true_pairs": total_true_pairs,
        "recalled_true_pairs": recalled_pairs,
    }
