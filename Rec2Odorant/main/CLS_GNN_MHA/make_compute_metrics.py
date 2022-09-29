import functools
import jax
from jax import numpy as jnp

from Rec2Odorant.main.metrics import confusion_matrix_binary

from Rec2Odorant.main.CLS_GNN_MHA.make_loss_func import make_loss_func


def make_compute_metrics(is_weighted, loss_option, num_thresholds = 200, use_jit = True):
    """
    Notes:
    ------
    If you want to jit compute_metrics set use_jit = True. This is necessary because jitted and non jitted versions use different 
    implementations of confusion_matrix. Non-jitted version uses sklearn.confusion_matrix and jitted version uses simplified version
    of sklearn.confusion_matrix with all of checks discareded and with jax.numpy instead of numpy.
    """
    thresholds = [(i + 1) * 1.0 / (num_thresholds - 1) for i in range(num_thresholds - 2)]
    loss_func = make_loss_func(is_weighted, option = loss_option)
    if use_jit:
        confusion_matrix = functools.partial(confusion_matrix_binary, labels = jnp.array([0,1]), sample_weight = None)
    else:
        from sklearn.metrics import confusion_matrix

    def _compute_metrics(logits, labels):
        # logits = jnp.squeeze(logits)
        logits = jnp.reshape(logits, newshape = (-1, )) # TODO: Check this
        labels = jnp.reshape(labels, newshape = (-1, )) # TODO: Check this
        pred_probs = jax.nn.sigmoid(logits)
        _C = {}
        conf_matrix = None
        for threshold in thresholds:
            _pred_labels = jnp.asarray(pred_probs > threshold).astype(jnp.int32)
            _C[threshold] = confusion_matrix(y_true = labels, y_pred = _pred_labels)
        pred_labels = jnp.round(pred_probs).astype(jnp.int32)
        conf_matrix = confusion_matrix(y_true = labels, y_pred = pred_labels)
        return conf_matrix, _C

    if is_weighted:
        def compute_metrics(logits, labels):
            loss = loss_func(logits, labels)
            labels, _ = labels
            conf_matrix, confusion_matrix_per_threshold = _compute_metrics(logits, labels)
            metrics = {'loss' : loss,
                    'confusion_matrix' : conf_matrix,
                    'confusion_matrix_per_threshold' : confusion_matrix_per_threshold,
                    }
            return metrics
    else:
        def compute_metrics(logits, labels):
            loss = loss_func(logits, labels) # This is already mean of all elements
            conf_matrix, confusion_matrix_per_threshold = _compute_metrics(logits, labels)
            metrics = {'loss' : loss,
                    'confusion_matrix' : conf_matrix,
                    'confusion_matrix_per_threshold' : confusion_matrix_per_threshold,
                    }
            return metrics

    # init_metrics_state = {'loss' : jnp.zeros(shape = ()),
    #                     'confusion_matrix' : jnp.zeros(shape = (2,2)),
    #                     'confusion_matrix_per_threshold' : {threshold : jnp.zeros(shape = (2,2)) for threshold in thresholds}}
    if use_jit:
        return jax.jit(compute_metrics)
    else:
        return compute_metrics


# def update_metrics_state(metrics_state, batch_metrics):
#     metrics_state = jax.tree_multimap(lambda x,y: x + y, metrics_state, batch_metrics)
#     return metrics_state
        