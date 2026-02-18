"""
Text-level evaluation metrics: ROUGE-L and token-level F1.

These metrics work on raw text strings rather than term sets,
making them more suitable for evaluating verbose LLM output
against terse gold standard annotations.
"""

import re


def _tokenize(text):
    """Normalize and tokenize text into lowercase word tokens."""
    if not text:
        return []
    text = text.lower()
    # Remove punctuation except hyphens within words
    text = re.sub(r"[^\w\s\-]", " ", text)
    tokens = text.split()
    # Filter very short tokens (single chars, numbers)
    return [t for t in tokens if len(t) > 1 and not t.isdigit()]


# ---------------------------------------------------------------------------
# ROUGE-L (Longest Common Subsequence)
# ---------------------------------------------------------------------------
def _lcs_length(x, y):
    """Compute length of Longest Common Subsequence of two sequences."""
    m, n = len(x), len(y)
    # Use 1D DP for memory efficiency
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def rouge_l(predicted_text, gold_text):
    """Compute ROUGE-L precision, recall, and F1.

    ROUGE-L uses the Longest Common Subsequence (LCS) between
    predicted and gold token sequences.

    Args:
        predicted_text: raw predicted string from LLM
        gold_text: gold standard string (joined terms)

    Returns: dict with precision, recall, f1
    """
    pred_tokens = _tokenize(predicted_text)
    gold_tokens = _tokenize(gold_text)

    if not pred_tokens and not gold_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_tokens or not gold_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs_len = _lcs_length(pred_tokens, gold_tokens)
    precision = lcs_len / len(pred_tokens)
    recall = lcs_len / len(gold_tokens)
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# Token-level F1 (SQuAD-style)
# ---------------------------------------------------------------------------
def token_f1(predicted_text, gold_text):
    """Compute token-level F1 (bag-of-words overlap).

    Treats both texts as bags of tokens and computes overlap-based
    precision, recall, and F1. This is the standard metric used in
    SQuAD and similar QA benchmarks.

    Args:
        predicted_text: raw predicted string from LLM
        gold_text: gold standard string (joined terms)

    Returns: dict with precision, recall, f1
    """
    pred_tokens = set(_tokenize(predicted_text))
    gold_tokens = set(_tokenize(gold_text))

    if not pred_tokens and not gold_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_tokens or not gold_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    shared = pred_tokens & gold_tokens
    precision = len(shared) / len(pred_tokens)
    recall = len(shared) / len(gold_tokens)
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {"precision": precision, "recall": recall, "f1": f1}


def compute_text_metrics(predicted_text, gold_text):
    """Compute all text-level metrics for a single prediction-gold pair.

    Returns: dict with rouge_l and token_f1 sub-dicts.
    """
    return {
        "rouge_l": rouge_l(predicted_text, gold_text),
        "token_f1": token_f1(predicted_text, gold_text),
    }
