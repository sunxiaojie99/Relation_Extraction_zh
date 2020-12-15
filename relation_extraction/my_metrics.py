from collections import Counter


def score(key, prediction):
    """Score function for f1,recall,precision ignoring no_relation."""
    NO_RELATION = "no_relation"
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation = Counter()

    # Loop over the data to compute a score
    for row in range(len(key)):
        gold = key[row]
        guess = prediction[row]

        if gold == NO_RELATION and guess == NO_RELATION:
            pass
        elif gold == NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
        elif gold != NO_RELATION and guess == NO_RELATION:
            gold_by_relation[gold] += 1
        elif gold != NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        tp = float(sum(correct_by_relation.values()))
        tp_fp = float(sum(guessed_by_relation.values()))
        prec_micro = tp / tp_fp
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        tp = float(sum(correct_by_relation.values()))
        tp_fn = float(sum(gold_by_relation.values()))
        recall_micro = tp / tp_fn
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro
        f1_micro = f1_micro / (prec_micro + recall_micro)
    print("Precision (micro): {:.3%}".format(prec_micro))
    print("   Recall (micro): {:.3%}".format(recall_micro))
    print("       F1 (micro): {:.3%}".format(f1_micro))
    return prec_micro, recall_micro, f1_micro
