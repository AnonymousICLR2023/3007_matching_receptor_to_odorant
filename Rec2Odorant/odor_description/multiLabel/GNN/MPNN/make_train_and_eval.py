import jax
from jax import numpy as jnp

from Rec2Odorant.odor_description.multiLabel.GNN.MPNN.make_loss_func import make_loss_func
from Rec2Odorant.odor_description.multiLabel.GNN.MPNN.make_compute_metrics import make_compute_metrics
from Rec2Odorant.odor_description.utils import tf_to_jax


def make_train_step(loss_func, init_rngs, reg_loss_func = None):
    if reg_loss_func is not None:
        # @jax.pmap
        @jax.jit
        def train_step(state, batch):
            """
            """
            def loss_fn(params):
                logits = state.apply_fn(params, batch[0], deterministic = False, rngs = state.rngs, return_embeddings=False)
                loss_val = loss_func(logits = logits, labels = batch[-1]) + reg_loss_func(params)
                return loss_val, logits
            grad_fn = jax.grad(loss_fn, has_aux = True)
            grads, logits = grad_fn(state.params)
            # grads = jax.lax.pmean(grads, axis_name='batch')
            # updates, opt_state = opt.update(updates = grads, state = state.opt_state)
            # params = optax.apply_updates(state.params, updates)
            state = state.apply_gradients(grads = grads) # This handles updates of opt_state and params
            return state, logits, grads
    else:
        # @jax.pmap
        @jax.jit
        def train_step(state, batch):
            """
            """
            def loss_fn(params):
                logits = state.apply_fn(params, batch[0], deterministic = False, rngs = state.rngs, return_embeddings=False)
                loss_val = loss_func(logits = logits, labels = batch[-1])
                return loss_val, logits
            grad_fn = jax.grad(loss_fn, has_aux = True)
            grads, logits = grad_fn(state.params)
            # grads = jax.lax.pmean(grads, axis_name='batch')
            # updates, opt_state = opt.update(updates = grads, state = state.opt_state)
            # params = optax.apply_updates(state.params, updates)
            state = state.apply_gradients(grads = grads) # This handles updates of opt_state and params
            return state, logits, grads
    return train_step


def make_eval_step():
    # @jax.pmap
    @jax.jit
    def eval_step(state, batch):
        logits = state.apply_fn(state.params, batch[0], deterministic = True, return_embeddings=False)
        return logits
    return eval_step

def make_train_epoch(is_weighted, num_classes, loss_option, init_rngs, logger, reg_loss_func = None, loader_output_type= 'jax'):
    """
    Helper function to create train_epoch function.
    """
    loss_func = make_loss_func(is_weighted = is_weighted, num_classes = num_classes, option = loss_option)
    compute_metrics = make_compute_metrics(is_weighted = is_weighted, num_classes = num_classes, loss_option = loss_option)
    train_step = make_train_step(loss_func = loss_func, init_rngs = init_rngs, reg_loss_func = reg_loss_func)
    if loader_output_type == 'jax':
        def train_epoch(state, loader):
            batch_metrics = []
            for i, batch in enumerate(loader):
                mols = batch[0]
                labels = batch[1]
                batch = (mols, labels)
                # batch = flax.jax_utils.replicate(batch)
                state = state.replace(rngs = jax.tree_map(lambda x: jax.random.split(x)[0], state.rngs))
                state, logits, grads = train_step(state, batch)
                metrics = compute_metrics(logits, labels = batch[-1])
                logger.debug('{}:  loss:  {}'.format(i, metrics['loss']))
                _loss = metrics['loss']
                if not _loss == _loss:
                    print(jax.tree_map(lambda x: jnp.max(jnp.abs(x)), grads))
                    print('Previous grads: -------------------')
                    print(jax.tree_map(lambda x: jnp.max(jnp.abs(x)), prev_grads))
                    print(_loss)
                    raise Exception('Loss is NaN')
                batch_metrics.append(metrics)
                prev_grads = grads
            loader.reset()
            return state, batch_metrics

    elif loader_output_type == 'tf':
        def train_epoch(state, loader):
            batch_metrics = []
            for i, batch in loader.enumerate():
                batch = jax.tree_map(lambda x: jax.device_put(tf_to_jax(x), device = jax.devices()[0]), batch)
                mols = batch[0]
                labels = batch[1]
                batch = (mols, labels)
                # batch = flax.jax_utils.replicate(batch)
                state = state.replace(rngs = jax.tree_map(lambda x: jax.random.split(x)[0], state.rngs)) # update PRNGKeys
                
                state, logits, grads = train_step(state, batch)
                metrics = compute_metrics(logits, labels = batch[1])
                logger.debug('{}:  loss:  {}'.format(i, metrics['loss']))
                _loss = metrics['loss']
                if not _loss == _loss:
                    print(jax.tree_map(lambda x: jnp.max(jnp.abs(x)), grads))
                    print('Previous grads: -------------------')
                    print(jax.tree_map(lambda x: jnp.max(jnp.abs(x)), prev_grads))
                    print(_loss)
                    raise Exception('Loss is NaN')
                batch_metrics.append(metrics)
                prev_grads = grads
            return state, batch_metrics
    return train_epoch


def make_valid_epoch(num_classes, loss_option, logger, loader_output_type = 'jax'):
    """
    Helper function to create valid_epoch function.
    """
    compute_metrics = make_compute_metrics(is_weighted = False, num_classes = num_classes, loss_option = loss_option)
    eval_step = make_eval_step()

    if loader_output_type == 'jax':
        def valid_epoch(state, valid_loader):
            batch_metrics = []
            for i, batch in enumerate(valid_loader):
                mols = batch[0]  
                labels = batch[1]
                batch = (mols, labels)
                # batch = flax.jax_utils.replicate(batch)
                logits = eval_step(state, batch)
                metrics = compute_metrics(logits, labels = labels)
                logger.debug('eval_step: {}:  eval_loss:  {}'.format(i, metrics['loss']))
                batch_metrics.append(metrics)
            valid_loader.reset()
            return batch_metrics

    elif loader_output_type == 'tf':
        def valid_epoch(state, valid_loader):
            batch_metrics = []
            for i, batch in valid_loader.enumerate():
                batch = jax.tree_map(lambda x: jax.device_put(tf_to_jax(x), device = jax.devices()[0]), batch)
                mols = batch[0]
                labels = batch[1]
                batch = (mols, labels) 
                # batch = flax.jax_utils.replicate(batch)
                logits = eval_step(state, batch)
                metrics = compute_metrics(logits, labels = labels)
                logger.debug('eval_step: {}:  eval_loss:  {}'.format(i, metrics['loss']))
                batch_metrics.append(metrics)
            return batch_metrics
    return valid_epoch