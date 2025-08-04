import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve


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


def plot_roc_curve(y_true, y_pred_probs):
    """plot the ROC curve and saves it to a file"""
    with sns.axes_style("whitegrid"):
        y_true_binary = (y_true != "Normal").astype(int)

        fpr, tpr, _ = roc_curve(y_true_binary, y_pred_probs)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label="ROC curve (area = %0.2f)" % roc_auc,
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plt.show()


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
