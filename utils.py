import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(trues, preds, model_name, class_names=['A', 'B', 'C', 'D', 'E'], save_path='confusion.png'):
    cm = confusion_matrix(trues, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='red' if i != j and cm[i, j] > 0 else 'black')

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    plt.title(f'{model_name} Confusion Matrix')
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()