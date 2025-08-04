import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve

from config import NON_CANCER_STATUS


def make_class_names(df):
    prob_cols = [col for col in df.columns if col.startswith("prob_")]
    class_names = sorted(
        [col.replace("prob_", "").replace("_", " ").title() for col in prob_cols]
    )
    return class_names


def calculate_sens_spec(y_true, y_pred, class_names):
    """calculate sensitivity and specificity for each class."""
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    results = {}

    for i, class_name in enumerate(class_names):
        tp = cm[i, i]
        fn = sum(cm[i, :]) - tp
        fp = sum(cm[:, i]) - tp
        tn = cm.sum() - (tp + fn + fp)

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        results[class_name] = {"sensitivity": sensitivity, "specificity": specificity}

    return pd.DataFrame.from_dict(results, orient="index")


def add_roc_curve_to_ax(y_true, y_pred_probs, label, ax=None):
    """
    calculate and plot a single ROC curve on a given matplotlib axes object.
    if no axes is provided, it creates a new figure and axes.

    Assume we have the results from two models:
    model1_probs = 1 - df1["prob_normal"]]
    model2_probs = 1 - df2["prob_normal"]
    y_true = df1["true_label"]

    fig, ax = add_roc_curve_to_ax(y_true, model1_probs, label="Model1")
    add_roc_curve_to_ax(y_true, model2_probs, label="Model2", ax=ax)
    plt.show()

    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    else:
        fig = ax.get_figure()

    y_true_binary = (y_true != NON_CANCER_STATUS).astype(int)
    fpr, tpr, _ = roc_curve(y_true_binary, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {roc_auc:.2f})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")

    return fig, ax


def plot_confusion_matrix(y_true, y_pred, class_names):
    """plot a confusion matrix using seaborn and saves it."""
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    with sns.axes_style("white"):
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title("Normalized Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.show()
