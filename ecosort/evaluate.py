import torch
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model, dataloader, class_names):
    """
    Evaluate a trained model on a test dataset.

    Returns:
        accuracy
        y_true
        y_pred
        confusion matrix
    """

    # DEVICE

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = model.to(device)
    model.eval()
    y_true = []
    y_pred = []
    y_confidence=[]
    y_alternatives=[]
    y_alternative_classes=[]

    # PREDICTIONS

    with torch.no_grad():

        for batch_feature, batch_label in dataloader:

            batch_feature = batch_feature.to(device)
            batch_label = batch_label.to(device)

            output = model(batch_feature)
            output=torch.softmax(output,dim=1)
            confidence_score,predictions=torch.topk(output,k=4)
            

            y_true.extend(
                batch_label.cpu().numpy()
            )

            y_confidence.extend(
                confidence_score[:,0].cpu().numpy()
            )

            y_alternatives.extend(
                confidence_score[:,1:].cpu().numpy()
            )

            y_alternative_classes.extend(
                predictions[:,1:].cpu().numpy()
            )
            y_pred.extend(
                predictions[:,0].cpu().numpy()
            )


    # ACCURACY

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    print(f"Test Accuracy: {accuracy:.4f}")


    # PRECISION / RECALL / F1

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=range(len(class_names)),
            zero_division=0
        )
    )


    print("\nPer-Class Metrics")
    print("------------------")

    for i, class_name in enumerate(class_names):

        print(
            f"{class_name:12s} | "
            f"Precision: {precision[i]:.4f} | "
            f"Recall: {recall[i]:.4f} | "
            f"F1: {f1[i]:.4f}"
        )


    # CLASSIFICATION REPORT

    print("\nClassification Report")
    print("---------------------")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=range(len(class_names)),
            target_names=class_names,
            zero_division=0
        )
    )


    # CONFUSION MATRIX

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=range(len(class_names))
    )


    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    plt.show()


    return accuracy, y_true, y_pred, y_confidence, y_alternatives, y_alternative_classes, cm