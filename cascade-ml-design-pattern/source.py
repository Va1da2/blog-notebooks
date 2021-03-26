import logging
import warnings
from typing import Optional, Union, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from imblearn.base import BaseSampler
from scipy.stats import spearmanr, kendalltau
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_recall_curve

plt.style.use('fivethirtyeight')

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
    display_top_f1_scores(
        precision,
        recall,
        f1score,
        thresholds,
        precision_threashold=kwrgs.get("precision_threashold"),
        recall_threshold=kwrgs.get("recall_threshold")
    )
    
    return cls, precision, recall, f1score


def display_top_f1_scores(
    precision: np.ndarray, 
    recall: np.ndarray, 
    f1score: np.ndarray, 
    threshold: np.ndarray, 
    precision_threashold: Optional[float]=None,
    recall_threshold: Optional[float]=None) -> None:
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
    if precision_threashold:
        df = df[df["Precision"] >= precision_threashold]
    if recall_threshold:
        df = df[df["Recall"] >= recall_threshold]
    
    display(df.head())


def run_algorithm_analysis(
    name: str,
    algorithm: BaseEstimator, 
    X_train: pd.DataFrame, 
    y_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    y_test: pd.DataFrame) -> Tuple[BaseEstimator, pd.DataFrame]:
    
    print_md(f"## {name}")
    
    model = algorithm().fit(X_train, y_train)
    if issubclass(algorithm, RegressorMixin):
        predictions = model.predict(X_test)
    elif issubclass(algorithm, ClassifierMixin):
        predictions = model.predict_proba(X_test)[:,-1]
    else:
        logger.error(f"Model is of unsupported type: {type(model)}")
        return
    
    results = pd.DataFrame(
        {
        "predicted": predictions,
        "actual": y_test,
        }
    )
    print()
    coef, p = spearmanr(results["predicted"], results["actual"])
    coef_k, p_k = kendalltau(results["predicted"], results["actual"])
    
    print_md("**Correlations between predicted and actual values for the Test Set**")
    print_md(f"""
    | Metric | Value | p-value |
    | ------ | ----- | ------- |
    | Spearman's Rho | {coef:.3f} | {p:.2E} |
    | Kendall's Tau | {coef_k:.3f} | {p_k:.2E} |
    """.strip())
    print()
    
    m, b = np.polyfit(y_test.to_list(), predictions, 1)
    
    plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    _ = plt.scatter(results["actual"].to_list(), results["predicted"].to_list())
    _ = plt.plot(results["actual"].to_numpy(), m*results["actual"].to_numpy() + b, color="r")
    _ = plt.title("Actual vs. Predicted Popularity", fontsize=18)
    _ = plt.xlabel("Actual", fontsize=14)
    _ = plt.ylabel("Predicted", fontsize=14)
    
    plt.show()
    
    return model, results

def print_md(string: str, color: Optional[str]=None) -> None:
    """https://stackoverflow.com/a/46934204/6602729"""
    colorstr = "<span style='color:{}'>{}</span>".format(color, string)
    display(Markdown(colorstr))


def return_class(score: Union[int, float], low: Union[int, float], high: Union[int, float]) -> float:
    if score <= low:
        return 0.
    elif score >= high:
        return 2.
    else:
        return 1.
