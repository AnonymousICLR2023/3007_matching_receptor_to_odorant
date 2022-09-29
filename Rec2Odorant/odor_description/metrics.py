import io
import itertools
import numpy as np
from matplotlib import pyplot as plt
from objax.jaxboard import Image
import jax
# from sklearn.metrics._plot.roc_curve import RocCurveDisplay
# from sklearn.metrics._classification import _prf_divide, _warn_prf
# from sklearn.metrics._ranking import auc


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
    -----
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes

    Ref:
    ----
    https://towardsdatascience.com/exploring-confusion-matrix-evolution-on-tensorboard-e66b39f4ac12
    """
    # Normalize the confusion matrix.
    # cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    figure = plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    # plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     color = "white" if cm[i, j] > threshold else "black"
    #     plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

def plot_to_png(figure):
    """
    Converts the matplotlib plot specified by 'figure' to a numpy array and
    returns it.
    """
    buf = io.BytesIO()    
    plt.savefig(buf, format='png')
    shape = (int(figure.bbox.bounds[3]), int(figure.bbox.bounds[2]), 4)

    plt.close(figure)
    buf.seek(0)

    image = buf.getvalue()
    return image, shape

def log_confusion_matrix(cm, class_names):
    figure = plot_confusion_matrix(cm, class_names)
    image, shape = plot_to_png(figure)
    return Image(shape, image)