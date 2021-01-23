import logging
from typing import Optional, Union, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from IPython.display import Markdown, display
from scipy.stats import spearmanr, kendalltau
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

plt.style.use('fivethirtyeight')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)


def run_algorithm_analysis(
    name: str,
    algorithm: BaseEstimator, 
    X_train: pd.DataFrame, 
    y_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    y_test: pd.DataFrame) -> Tuple[BaseEstimator, pd.DataFrame]:
    
    display(Markdown(f"## {name}"))
    
    model = algorithm().fit(X_train, y_train)
    if issubclass(algorithm, RegressorMixin):
        predictions = model.predict(X_test)
    elif issubclass(algorithm, ClassifierMixin):
        predictions = model.predict_proba(X_test)[:,-1]
    else:
        logger.error(f"mMolde is of unsupported type: {type(model)}")
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
    display(Markdown(f"""
    | Metric | Value | p-value |
    | ------ | ----- | ------- |
    | Spearman's Rho | {coef:.3f} | {p:.2E} |
    | Kendall's Tau | {coef_k:.3f} | {p_k:.2E} |
    """.strip()))
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
