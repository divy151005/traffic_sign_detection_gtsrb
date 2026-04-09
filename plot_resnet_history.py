
"""
Plotting utilities for ResNetV3 training history and evaluation.

Usage:
1. After training in your resnetV3 notebook, ensure you have `history` object:
   history = model.fit(...)

2. Paste and run these functions in a new code cell to visualize training curves and (optionally)
   confusion matrix / classification report.

Example:
    plot_training_history(history)
    plot_confusion_matrix_from_model(model, X_test, y_test, class_names)

"""

import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def plot_training_history(history, figsize=(10,4)):
    """
    Plots training & validation accuracy and loss from a Keras `History` object.
    Handles common key names ('accuracy'/'acc').
    """
    hist = history.history if not isinstance(history, dict) else history
    # detect accuracy key
    acc_key = None
    val_acc_key = None
    for k in hist.keys():
        if k.lower().endswith('accuracy') or k.lower().endswith('acc'):
            if not k.lower().startswith('val'):
                acc_key = k
            else:
                val_acc_key = k
    # detect val keys
    if val_acc_key is None:
        val_acc_key = 'val_' + acc_key if acc_key else None

    loss_key = None
    val_loss_key = None
    for k in hist.keys():
        if 'loss' in k.lower() and not k.lower().startswith('val'):
            loss_key = k
        if 'val_loss' == k.lower():
            val_loss_key = k
    if val_loss_key is None and loss_key:
        val_loss_key = 'val_' + loss_key

    epochs = range(1, len(next(iter(hist.values()))) + 1)

    plt.figure(figsize=(figsize[0], figsize[1]))
    # Accuracy
    plt.subplot(1,2,1)
    if acc_key and acc_key in hist:
        plt.plot(epochs, hist[acc_key], label='Train')
    if val_acc_key and val_acc_key in hist:
        plt.plot(epochs, hist[val_acc_key], label='Val')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Loss
    plt.subplot(1,2,2)
    if loss_key and loss_key in hist:
        plt.plot(epochs, hist[loss_key], label='Train')
    if val_loss_key and val_loss_key in hist:
        plt.plot(epochs, hist[val_loss_key], label='Val')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix_from_model(model, X_test, y_test, class_names=None,
                                     normalize=False, figsize=(8,8), cmap=None):
    """
    Predicts labels using model and plots confusion matrix.
    - model: Keras model with predict method
    - X_test: test images (numpy array)
    - y_test: true labels (one-hot or integer encoded)
    - class_names: list of class names (optional)
    """
    # get true labels as integers
    if y_test.ndim > 1 and y_test.shape[1] > 1:
        y_true = np.argmax(y_test, axis=1)
    else:
        y_true = y_test.ravel().astype(int)

    y_prob = model.predict(X_test, verbose=0)
    if y_prob.ndim > 1 and y_prob.shape[1] > 1:
        y_pred = np.argmax(y_prob, axis=1)
    else:
        y_pred = (y_prob > 0.5).astype(int).ravel()

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)

    plt.figure(figsize=figsize)
    if cmap is None:
        # use seaborn's default palette (no explicit color passed)
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                    xticklabels=class_names, yticklabels=class_names, square=True)
    else:
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.colorbar()
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], '.2f' if normalize else 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        if class_names is not None:
            plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
            plt.yticks(range(len(class_names)), class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix' + (' (normalized)' if normalize else ''))
    plt.tight_layout()
    plt.show()

def print_classification_report_from_model(model, X_test, y_test, class_names=None):
    """
    Prints sklearn classification report comparing model predictions to true labels.
    """
    if y_test.ndim > 1 and y_test.shape[1] > 1:
        y_true = np.argmax(y_test, axis=1)
    else:
        y_true = y_test.ravel().astype(int)

    y_prob = model.predict(X_test, verbose=0)
    if y_prob.ndim > 1 and y_prob.shape[1] > 1:
        y_pred = np.argmax(y_prob, axis=1)
    else:
        y_pred = (y_prob > 0.5).astype(int).ravel()

    print(classification_report(y_true, y_pred, target_names=class_names))
