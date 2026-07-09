import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate_model(y_true, y_prob):
    try:
        roc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc = 0.5
        
    try:
        pr = average_precision_score(y_true, y_prob)
    except ValueError:
        pr = 0.0
    
    if len(y_prob) > 0 and y_true.sum() > 0:
        k = int(len(y_prob) * 0.10)
        if k < 1: k = 1
        
        top_idx = np.argsort(-y_prob)[:k]
        
        tp_top = y_true[top_idx].sum()
        
        recall_top = tp_top / y_true.sum()
        
        precision_top = tp_top / k
    else:
        recall_top = 0.0
        precision_top = 0.0
    
    return roc, pr, recall_top, precision_top

def bootstrap_auc(y_true, y_prob, n=1000, seed=42):
    rng = np.random.RandomState(seed)
    scores = []
    
    if len(np.unique(y_true)) < 2:
        return 0.0, 0.0

    for _ in range(n):
        
        indices = rng.choice(len(y_true), len(y_true), replace=True)
        y_true_sample = y_true[indices]
        y_prob_sample = y_prob[indices]
        
        if len(np.unique(y_true_sample)) < 2:
            continue
            
        try:
            score = roc_auc_score(y_true_sample, y_prob_sample)
            scores.append(score)
        except ValueError:
            pass
            
    if not scores:
        return 0.0, 0.0
        
    return np.mean(scores), np.std(scores)