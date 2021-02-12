import logging
import warnings
from typing import Optional, Union, Tuple

import pandas as pd
import numpy as np
from IPython.display import display
from imblearn.base import BaseSampler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_recall_curve


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)


def run_single_experiment(data_augmentation_class: BaseSampler, 
                          X_train: pd.DataFrame, 
                          y_train: pd.DataFrame, 
                          X_test: pd.DataFrame,
                          y_test: pd.DataFrame, 
                          **kwrgs) -> Tuple[GradientBoostingClassifier, np.ndarray, np.ndarray, np.ndarray]:
    """Runs a single experiment for a given data augmentation method.
    """
    X_train_resampled, y_train_resampled = data_augmentation_class.fit_resample(X_train, y_train)
    cls = GradientBoostingClassifier().fit(X_train_resampled, y_train_resampled)
    preds = cls.predict_proba(X_test)[:, 1]
    
    precision, recall, thresholds = precision_recall_curve(y_test, preds)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f1score = 2 * (precision * recall) / (precision + recall)
    display_top_f1_scores(precision, recall, f1score, thresholds)
    
    return cls, precision, recall, f1score


def display_top_f1_scores(precision: np.ndarray, recall: np.ndarray, f1score: np.ndarray, threshold: np.ndarray) -> None:
    """Displays a best performing model threshold with corresponding metrics: precision, recall, and f1 score.
    """
    try:
        assert len(precision) == len(recall) == len(f1score) == len(threshold)+1
    except:
        logger.error(f"precision: {len(precision)}, recall: {len(recall)}, f1: {len(f1score)}, threshold: {len(threshold)}")
        raise
    df = pd.DataFrame({
        "Precision": precision[:-1],
        "Recall": recall[:-1],
        "F1-score": f1score[:-1],
        "Threshold": threshold
    })
    df = df.sort_values(by="F1-score", ascending=False)
    display(df.head())
