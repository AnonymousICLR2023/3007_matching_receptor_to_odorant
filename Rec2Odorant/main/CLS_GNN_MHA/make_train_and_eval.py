import jax
from jax import numpy as jnp

from Rec2Odorant.main.utils import tf_to_jax

from Rec2Odorant.main.CLS_GNN_MHA.make_loss_func import make_loss_func
from Rec2Odorant.main.CLS_GNN_MHA.make_compute_metrics import make_compute_metrics

def make_train_step(loss_func, init_rngs, reg_loss_func = None):
    if reg_loss_func is not None:
        def train_step(state, batch):
            """
            """
            state = state.replace(rngs = jax.tree_map(lambda x: jax.random.split(x)[0], state.rngs)) # update PRNGKeys
            def loss_fn(params):
                logits = state.apply_fn(params, batch[:-1], deterministic = False, rngs = state.rngs) # TODO init_rngs ???
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
        def train_step(state, batch):
            """
            """
            state = state.replace(rngs = jax.tree_map(lambda x: jax.random.split(x)[0], state.rngs)) # update PRNGKeys
            def loss_fn(params):
                logits = state.apply_fn(params, batch[:-1], deterministic = False, rngs = state.rngs)
                loss_val = loss_func(logits = logits, labels = batch[-1])
                return loss_val, logits
            grad_fn = jax.grad(loss_fn, has_aux = True)
            grads, logits = grad_fn(state.params)
            # grads = jax.lax.pmean(grads, axis_name='batch')
            # updates, opt_state = opt.update(updates = grads, state = state.opt_state)
            # params = optax.apply_updates(state.params, updates)
            state = state.apply_gradients(grads = grads) # This handles updates of opt_state and params
            return state, logits, grads
    # return train_step
    return jax.jit(train_step)


def make_eval_step():
    def eval_step(state, batch):
        logits = state.apply_fn(state.params, batch[:-1], deterministic = True)
        return logits
    # return eval_step
    return jax.jit(eval_step)


def make_train_epoch(is_weighted, loss_option, init_rngs, logger, reg_loss_func = None, loader_output_type = 'jax'):
    """
    Helper function to create train_epoch function.
    """
    loss_func = make_loss_func(is_weighted = is_weighted, option = loss_option)
    compute_metrics = make_compute_metrics(is_weighted = is_weighted, loss_option = loss_option, use_jit = True)
    train_step = make_train_step(loss_func = loss_func, init_rngs = init_rngs, reg_loss_func = reg_loss_func)
    # Case loader outputs jnp.DeviceArray:
    if loader_output_type == 'jax':
        def train_epoch(state, loader):
            batch_metrics = []
            for i, batch in enumerate(loader):
                seq = batch[0]
                G = batch[1] # mols, line_mols
                labels = batch[2]
                S = seq # ['hidden_states']
                batch = (S, G, labels)
                # batch = (S, mols, line_mols, labels)
                # batch = flax.jax_utils.replicate(batch)
                # state = state.replace(rngs = jax.tree_map(lambda x: jax.random.split(x)[0], state.rngs)) # update PRNGKeys
                state, logits, _ = train_step(state, batch)
                metrics = compute_metrics(logits, labels = batch[-1])
                # logger.debug('{}:  loss:  {}'.format(i, metrics['loss']))
                # _loss = metrics['loss']
                # if not _loss == _loss:
                #     print(jax.tree_map(lambda x: jnp.max(jnp.abs(x)), grads))
                #     print('Previous grads: -------------------')
                #     print(jax.tree_map(lambda x: jnp.max(jnp.abs(x)), prev_grads))
                #     print(_loss)
                #     raise Exception('Loss is NaN')
                # prev_grads = grads
                batch_metrics.append(metrics)
            loader.reset()
            return state, batch_metrics
    # Case loader outputs tf.Tensor:
    elif loader_output_type == 'tf':
        def train_epoch(state, loader):
            batch_metrics = []
            for i, batch in loader.enumerate():
                batch = jax.tree_map(lambda x: jax.device_put(tf_to_jax(x), device = jax.devices()[0]), batch)
                seq = batch[0]
                G = batch[1] # mols, line_mols
                labels = batch[2]
                S = seq # ['hidden_states']
                batch = (S, G, labels)
                # batch = (S, mols, line_mols, labels)
                # batch = flax.jax_utils.replicate(batch)
                # state = state.replace(rngs = jax.tree_map(lambda x: jax.random.split(x)[0], state.rngs)) # update PRNGKeys
                state, logits, _ = train_step(state, batch)
                metrics = compute_metrics(logits, labels = batch[-1])
                # logger.debug('{}:  loss:  {}'.format(i, metrics['loss']))
                # _loss = metrics['loss']
                # if not _loss == _loss:
                #     print(jax.tree_map(lambda x: jnp.max(jnp.abs(x)), grads))
                #     print('Previous grads: -------------------')
                #     print(jax.tree_map(lambda x: jnp.max(jnp.abs(x)), prev_grads))
                #     print(_loss)
                #     raise Exception('Loss is NaN')
                # prev_grads = grads
                batch_metrics.append(metrics)
            return state, batch_metrics

    return train_epoch


def make_valid_epoch(loss_option, logger, loader_output_type = 'jax'):
    """
    Helper function to create valid_epoch function.
    """
    compute_metrics = make_compute_metrics(is_weighted = False, loss_option = loss_option, use_jit = True)
    eval_step = make_eval_step()
    # Case loader outputs jnp.DeviceArray:
    if loader_output_type == 'jax':
        def valid_epoch(state, valid_loader):
            batch_metrics = []
            for i, batch in enumerate(valid_loader):
                seq = batch[0]
                # mols, line_mols = batch[1]
                G = batch[1]
                labels = batch[2]
                if isinstance(labels, (list, tuple)):
                    labels = labels[0]
                S = seq # ['hidden_states']
                batch = (S, G, labels)
                # batch = (S, mols, line_mols, labels)
                # batch = flax.jax_utils.replicate(batch)
                logits = eval_step(state, batch)
                metrics = compute_metrics(logits, labels = labels)
                logger.debug('eval_step: {}:  eval_loss:  {}'.format(i, metrics['loss']))
                batch_metrics.append(metrics)
            valid_loader.reset()
            return batch_metrics
    # Case loader outputs tf.Tensor:
    elif loader_output_type == 'tf':
        def valid_epoch(state, valid_loader):
            batch_metrics = []
            for i, batch in valid_loader.enumerate():
                batch = jax.tree_map(lambda x: jax.device_put(tf_to_jax(x), device = jax.devices()[0]), batch)
                seq = batch[0]
                # mols, line_mols = batch[1]
                G = batch[1] 
                labels = batch[2]
                if isinstance(labels, (list, tuple)):
                    labels = labels[0]
                S = seq # ['hidden_states']
                batch = (S, G, labels)
                # batch = (S, mols, line_mols, labels)
                # batch = flax.jax_utils.replicate(batch)
                logits = eval_step(state, batch)
                metrics = compute_metrics(logits, labels = labels)
                logger.debug('eval_step: {}:  eval_loss:  {}'.format(i, metrics['loss']))
                batch_metrics.append(metrics)
            return batch_metrics
    return valid_epoch




# --------
# jax.pmap
# --------
def make_train_step_pmap(loss_func, init_rngs, reg_loss_func = None):
    if reg_loss_func is not None:
        def train_step(state, batch):
            """
            """
            state = state.replace(rngs = jax.tree_map(lambda x: jax.random.split(x)[0], state.rngs)) # update PRNGKeys
            def loss_fn(params):
                logits = state.apply_fn(params, batch[:-1], deterministic = False, rngs = state.rngs)
                loss_val = loss_func(logits = logits, labels = batch[-1]) + reg_loss_func(params)
                return loss_val, logits
            grad_fn = jax.grad(loss_fn, has_aux = True)
            grads, logits = grad_fn(state.params)
            grads = jax.lax.pmean(grads, axis_name='batch')
            # updates, opt_state = opt.update(updates = grads, state = state.opt_state)
            # params = optax.apply_updates(state.params, updates)
            state = state.apply_gradients(grads = grads) # This handles updates of opt_state and params
            return state, logits, grads
    else:
        def train_step(state, batch):
            """
            """
            state = state.replace(rngs = jax.tree_map(lambda x: jax.random.split(x)[0], state.rngs)) # update PRNGKeys
            def loss_fn(params):
                logits = state.apply_fn(params, batch[:-1], deterministic = False, rngs = state.rngs)
                loss_val = loss_func(logits = logits, labels = batch[-1])
                return loss_val, logits
            grad_fn = jax.grad(loss_fn, has_aux = True)
            grads, logits = grad_fn(state.params)
            grads = jax.lax.pmean(grads, axis_name='batch')
            # updates, opt_state = opt.update(updates = grads, state = state.opt_state)
            # params = optax.apply_updates(state.params, updates)
            state = state.apply_gradients(grads = grads) # This handles updates of opt_state and params
            return state, logits, grads
    return jax.pmap(train_step, axis_name='batch')


def make_eval_step_pmap():
    def eval_step(state, batch):
        logits = state.apply_fn(state.params, batch[:-1], deterministic = True)
        return logits
    return jax.pmap(eval_step, axis_name='batch')


def make_train_epoch_pmap(is_weighted, loss_option, init_rngs, logger, reg_loss_func = None, loader_output_type = 'jax'):
    """
    Helper function to create train_epoch function.
    """
    raise NotImplementedError('batch = (S, G, labels) needs to be changed probably')
    loss_func = make_loss_func(is_weighted = is_weighted, option = loss_option)
    compute_metrics = make_compute_metrics(is_weighted = is_weighted, loss_option = loss_option, use_jit = True)
    train_step = make_train_step_pmap(loss_func = loss_func, init_rngs = init_rngs, reg_loss_func = reg_loss_func)
    if loader_output_type == 'jax':
        def train_epoch(state, loader):
            batch_metrics = []
            for i, batch in enumerate(loader):
                seq = batch[0]
                mols, line_mols = batch[1]
                labels = batch[2]
                S = seq # ['hidden_states']
                batch = (S, mols, line_mols, labels)
                # print(jax.tree_map(lambda x: x.shape, batch))
                # print('------')
                # batch = flax.jax_utils.replicate(batch)
                # print(state.rngs) 
                state, logits, grads = train_step(state, batch)
                metrics = compute_metrics(logits, labels = batch[-1])
                logger.debug('{}:  loss:  {}'.format(i, metrics['loss']))
                # _loss = metrics['loss']
                # if not _loss == _loss:
                #     print(jax.tree_map(lambda x: jnp.max(jnp.abs(x)), grads))
                #     print('Previous grads: -------------------')
                #     print(jax.tree_map(lambda x: jnp.max(jnp.abs(x)), prev_grads))
                #     print(_loss)
                #     raise Exception('Loss is NaN')
                # prev_grads = grads
                batch_metrics.append(metrics)
            return state, batch_metrics
    elif loader_output_type == 'tf':
        def train_epoch(state, loader):
            batch_metrics = []
            for i, batch in loader.enumerate():
                batch = jax.tree_map(lambda x: jax.device_put(tf_to_jax(x), device = jax.devices()[0]), batch)
                seq = batch[0]
                mols, line_mols = batch[1]
                labels = batch[2]
                S = seq # ['hidden_states']
                batch = (S, mols, line_mols, labels)
                # print(jax.tree_map(lambda x: x.shape, batch))
                # print('------')
                # batch = flax.jax_utils.replicate(batch)
                state, logits, grads = train_step(state, batch)
                metrics = compute_metrics(logits, labels = batch[-1])
                logger.debug('{}:  loss:  {}'.format(i, metrics['loss']))
                # _loss = metrics['loss']
                # if not _loss == _loss:
                #     print(jax.tree_map(lambda x: jnp.max(jnp.abs(x)), grads))
                #     print('Previous grads: -------------------')
                #     print(jax.tree_map(lambda x: jnp.max(jnp.abs(x)), prev_grads))
                #     print(_loss)
                #     raise Exception('Loss is NaN')
                # prev_grads = grads
                batch_metrics.append(metrics)
            return state, batch_metrics
    return train_epoch


def make_valid_epoch_pmap(loss_option, logger, loader_output_type = 'jax'):
    """
    Helper function to create valid_epoch function.
    """
    raise NotImplementedError('batch = (S, G, labels) needs to be changed probably')
    compute_metrics = make_compute_metrics(is_weighted = False, loss_option = loss_option, use_jit = True)
    eval_step = make_eval_step_pmap()

    if loader_output_type == 'jax':
        def valid_epoch(state, valid_loader):
            batch_metrics = []
            for i, batch in enumerate(valid_loader):
                seq = batch[0]
                mols, line_mols = batch[1] 
                labels = batch[2]
                if isinstance(labels, (list, tuple)):
                    labels = labels[0]
                S = seq # ['hidden_states']
                batch = (S, mols, line_mols, labels)
                # batch = flax.jax_utils.replicate(batch)
                logits = eval_step(state, batch)
                metrics = compute_metrics(logits, labels = labels)
                logger.debug('eval_step: {}:  eval_loss:  {}'.format(i, metrics['loss']))
                batch_metrics.append(metrics)
            return batch_metrics
    elif loader_output_type == 'tf':
        def valid_epoch(state, valid_loader):
            batch_metrics = []
            for i, batch in valid_loader.enumerate():
                batch = jax.tree_map(lambda x: jax.device_put(tf_to_jax(x), device = jax.devices()[0]), batch)
                seq = batch[0]
                mols, line_mols = batch[1] 
                labels = batch[2]
                if isinstance(labels, (list, tuple)):
                    labels = labels[0]
                S = seq # ['hidden_states']
                batch = (S, mols, line_mols, labels)
                # batch = flax.jax_utils.replicate(batch)
                logits = eval_step(state, batch)
                metrics = compute_metrics(logits, labels = labels)
                logger.debug('eval_step: {}:  eval_loss:  {}'.format(i, metrics['loss']))
                batch_metrics.append(metrics)
            return batch_metrics
    return valid_epoch