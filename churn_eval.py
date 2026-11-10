
import numpy as np
from sklearn.metrics import precision_recall_curve


def find_best_threshold(model, x_val, y_val):
  

    y_scores = model.predict_proba(x_val)[:, 1]

    precision, recall, thresholds = precision_recall_curve(
        y_val,
        y_scores
    )

    
    f1_scores = (
        2 * precision[:-1] * recall[:-1]
    ) / (
        precision[:-1] + recall[:-1] + 1e-10
    )

    best_index = np.argmax(f1_scores)

    best_threshold = thresholds[best_index]
    best_f1 = f1_scores[best_index]

    return best_threshold, best_f1
