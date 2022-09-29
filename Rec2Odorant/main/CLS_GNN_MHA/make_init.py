import jax
from jax import numpy as jnp
import jraph
import time
import tensorflow as tf

from Rec2Odorant.main.utils import create_line_graph_and_pad, pad_graph

def make_init_model(model, batch_size, n_partitions = 0, seq_embedding_size = 1024, num_node_features = 8, num_edge_features = 2, self_loops = False, line_graph = True):
    """
    Parameters:
    -----------
    seq_embedding_size : int
        size of embedding. 1024 for ProtBERT, 768 for TapeBERT
    """
    key = jax.random.PRNGKey(int(time.time()))
    # @jax.pmap
    # @jax.jit
    def init_model(rngs):
        # num_node_features = 8
        # num_edge_features = 3

        padding_n_node = 32
        padding_n_edge = 64

        # seq_embedding_size = 1024
        key_nodes, key_edges, key_S = jax.random.split(key, 3)
        num_nodes = jax.random.randint(key_nodes, minval=10, maxval=padding_n_node - 1, shape = ())
        num_edges = jax.random.randint(key_edges, minval=30, maxval=padding_n_edge - 1, shape = ())
        
        S = jax.random.normal(key = key_S, shape = (batch_size, 5 * seq_embedding_size))

        _mol = jraph.GraphsTuple(nodes = jnp.ones(shape = (num_nodes, num_node_features), dtype = jnp.float32),
                                edges = jnp.ones(shape = (num_edges, num_edge_features), dtype = jnp.float32),
                                receivers = jnp.concatenate([jnp.arange(num_edges -1) + 1, jnp.array([0], dtype=jnp.int32)]),  # circle
                                senders = jnp.arange(num_edges),                                                               # circle
                                n_node = jnp.array([num_nodes]),
                                n_edge = jnp.array([num_edges]),
                                globals = None,
                                )

        if self_loops:
            senders = jnp.concatenate([jnp.arange(num_nodes), _mol.senders])
            receivers = jnp.concatenate([jnp.arange(num_nodes), _mol.receivers])
            _mol = _mol._replace(edges = None, receivers = receivers, senders = senders)

        batch = [_mol for _ in range(batch_size)]

        if not line_graph:
            mols = []
            for mol in batch:
                padded_mol = pad_graph(mol, 
                                        padding_n_node = padding_n_node, 
                                        padding_n_edge = padding_n_edge,
                                        )
                mols.append(padded_mol)
            if n_partitions > 0:
                partition_size = len(batch) // n_partitions        
                mols = [jraph.batch(mols[i*partition_size:(i+1)*partition_size]) for i in range(n_partitions)]
            else:
                mols = jraph.batch(mols)
            G = mols
        else:
            mols = []
            line_mols = []
            for mol in batch:
                padded_mol, padded_line_mol = create_line_graph_and_pad(mol, 
                                                                        padding_n_node = padding_n_node, 
                                                                        padding_n_edge = padding_n_edge,
                                                                        )
                mols.append(padded_mol)
                line_mols.append(padded_line_mol)
            if n_partitions > 0:
                partition_size = len(batch) // n_partitions        
                mols, line_mols = [(jraph.batch(mols[i*partition_size:(i+1)*partition_size]), 
                                    jraph.batch(line_mols[i*partition_size:(i+1)*partition_size])) for i in range(n_partitions)]
            else:
                mols, line_mols = (jraph.batch(mols), jraph.batch(line_mols))
            G = (mols, line_mols)

        params = model.init(rngs, (S, G), deterministic = False)
        return params
    return init_model


# ---------
# TF specs:
# ---------
def get_tf_specs(n_partitions = 0, seq_embedding_size = 1024, is_weighted = True, num_node_features = 8, num_edge_features = 2, padding_n_node = 32, padding_n_edge = 64, line_graph = True, self_loops = False):
    """
    Get TensorSpec for tf.data.Dataset. These specs correspond to the output of the loader.
    """
    # num_node_features = 8
    # num_edge_features = 3

    # padding_n_node = 32
    # padding_n_edge = 64

    if self_loops:
        _G = _get_tf_specs_graph_self_loops(n_partitions = n_partitions, num_node_features = num_node_features, padding_n_node = padding_n_node, padding_n_edge = padding_n_edge)
        G = _G
    else:
        _G = _get_tf_specs_graph(n_partitions = n_partitions, num_node_features = num_node_features, num_edge_features = num_edge_features, padding_n_node = padding_n_node, padding_n_edge = padding_n_edge)
        if line_graph:
            _line_G = _get_tf_specs_line_graph(n_partitions = n_partitions, num_node_features = num_node_features, num_edge_features = num_edge_features, padding_n_edge = padding_n_edge)
            G = (_G, _line_G)
        else:
            G = _G

    if n_partitions == 0:
        S = tf.TensorSpec(shape=(None, 5*seq_embedding_size), dtype=tf.float32) # 3840
        if is_weighted:
            labels =(tf.TensorSpec(shape=(None, ), dtype=tf.int32),
                    tf.TensorSpec(shape=(None, ), dtype=tf.float32))
        else:
            labels = tf.TensorSpec(shape=(None, ), dtype=tf.int32)
    elif n_partitions > 0:
        S = tf.TensorSpec(shape=(n_partitions, None, 5*seq_embedding_size), dtype=tf.float32) # 3840
        if is_weighted:
            labels =(tf.TensorSpec(shape=(n_partitions, None, ), dtype=tf.int32),
                    tf.TensorSpec(shape=(n_partitions, None, ), dtype=tf.float32))
        else:
            labels = tf.TensorSpec(shape=(n_partitions, None, ), dtype=tf.int32)
    
    return (S, G, labels)


def _get_tf_specs_graph(n_partitions = 0, num_node_features = 8, num_edge_features = 2, padding_n_node = 32, padding_n_edge = 64):
    """
    Get TensorSpec for tf.data.Dataset. These specs correspond to the output of the loader.
    """
    if n_partitions == 0:
        return jraph.GraphsTuple(nodes     =   tf.TensorSpec(shape=(None, num_node_features), dtype=tf.float32), # (32*batch_size, 8)
                            edges        =   tf.TensorSpec(shape=(None, num_edge_features), dtype=tf.float32), # (64*batch_size, 2),
                            receivers    =   tf.TensorSpec(shape=(None, ),                  dtype=tf.int32),    # (64*batch_size,), 
                            senders      =   tf.TensorSpec(shape=(None, ),                  dtype=tf.int32), # (64*batch_size,), 
                            # globals      =   tf.TensorSpec(shape=(None, padding_n_node),    dtype=tf.bool), # (batch_size, 32),
                            globals      =  {'node_padding_mask' : tf.TensorSpec(shape=(None, padding_n_node),    dtype=tf.bool),
                                             'edge_padding_mask' : tf.TensorSpec(shape=(None, padding_n_edge),    dtype=tf.bool)}, # (batch_size, 32),  
                            n_node       =   tf.TensorSpec(shape=(None, ),                  dtype=tf.int32), # (2*batch_size,), 
                            n_edge       =   tf.TensorSpec(shape=(None, ),                  dtype=tf.int32)) # (2*batch_size,)),
    elif n_partitions > 0:
        return jraph.GraphsTuple(nodes     =   tf.TensorSpec(shape=(n_partitions, None, num_node_features), dtype=tf.float32), # (32*batch_size, 8)
                            edges        =   tf.TensorSpec(shape=(n_partitions, None, num_edge_features), dtype=tf.float32), # (64*batch_size, 2), 
                            receivers    =   tf.TensorSpec(shape=(n_partitions, None, ),                  dtype=tf.int32),    # (64*batch_size,), 
                            senders      =   tf.TensorSpec(shape=(n_partitions, None, ),                  dtype=tf.int32), # (64*batch_size,), 
                            # globals      =   tf.TensorSpec(shape=(n_partitions, None, padding_n_node),    dtype=tf.bool), # (batch_size, 32), 
                            globals      =  {'node_padding_mask' : tf.TensorSpec(shape=(n_partitions, None, padding_n_node),    dtype=tf.bool),
                                             'edge_padding_mask' : tf.TensorSpec(shape=(n_partitions, None, padding_n_edge),    dtype=tf.bool)}, # (batch_size, 32), 
                            n_node       =   tf.TensorSpec(shape=(n_partitions, None, ),                  dtype=tf.int32), # (2*batch_size,), 
                            n_edge       =   tf.TensorSpec(shape=(n_partitions, None, ),                  dtype=tf.int32)) # (2*batch_size,)),


def _get_tf_specs_line_graph(n_partitions = 0, num_node_features = 8, num_edge_features = 2, padding_n_edge = 64):
    """
    Get TensorSpec for tf.data.Dataset. These specs correspond to the output of the loader.
    """
    if n_partitions == 0:
        return jraph.GraphsTuple(nodes     =   tf.TensorSpec(shape=(None, num_edge_features),  dtype=tf.float32), # (64*batch_size, 2), 
                            edges       =   tf.TensorSpec(shape=(None, num_node_features),  dtype=tf.float32), # (32*9*batch_size, 8), 
                            receivers   =   tf.TensorSpec(shape=(None, ),                   dtype=tf.int32), # (32*9*batch_size,), 
                            senders     =   tf.TensorSpec(shape=(None, ),                   dtype=tf.int32), # (32*9*batch_size,), 
                            # globals     =   tf.TensorSpec(shape=(None, padding_n_edge),     dtype=tf.bool), # (batch_size, 64),
                            globals     =   tf.TensorSpec(shape=None                                      ), # (batch_size, 64), 
                            n_node      =   tf.TensorSpec(shape=(None, ),                   dtype=tf.int32), # (2*batch_size,), 
                            n_edge      =   tf.TensorSpec(shape=(None, ),                   dtype=tf.int32)) # (2*batch_size,))
    elif n_partitions > 0:
        return jraph.GraphsTuple(nodes     =   tf.TensorSpec(shape=(n_partitions, None, num_edge_features),  dtype=tf.float32), # (64*batch_size, 2), 
                            edges       =   tf.TensorSpec(shape=(n_partitions, None, num_node_features),  dtype=tf.float32), # (32*9*batch_size, 8), 
                            receivers   =   tf.TensorSpec(shape=(n_partitions, None, ),                   dtype=tf.int32), # (32*9*batch_size,), 
                            senders     =   tf.TensorSpec(shape=(n_partitions, None, ),                   dtype=tf.int32), # (32*9*batch_size,), 
                            # globals     =   tf.TensorSpec(shape=(n_partitions, None, padding_n_edge),     dtype=tf.bool), # (batch_size, 64),
                            globals     =   tf.TensorSpec(shape=None), # (batch_size, 64),  
                            n_node      =   tf.TensorSpec(shape=(n_partitions, None, ),                   dtype=tf.int32), # (2*batch_size,), 
                            n_edge      =   tf.TensorSpec(shape=(n_partitions, None, ),                   dtype=tf.int32)) # (2*batch_size,))


def _get_tf_specs_graph_self_loops(n_partitions = 0, num_node_features = 8, padding_n_node = 32, padding_n_edge = 96):
    """
    Get TensorSpec for tf.data.Dataset. These specs correspond to the output of the loader.
    """
    if n_partitions == 0:
        return jraph.GraphsTuple(nodes     =   tf.TensorSpec(shape=(None, num_node_features), dtype=tf.float32), # (32*batch_size, 8)
                            edges        =   tf.TensorSpec(shape=None                                     ),    # No edges allowed
                            receivers    =   tf.TensorSpec(shape=(None, ),                  dtype=tf.int32),    # (64*batch_size,), 
                            senders      =   tf.TensorSpec(shape=(None, ),                  dtype=tf.int32), # (64*batch_size,), 
                            # globals      =   tf.TensorSpec(shape=(None, padding_n_node),    dtype=tf.bool), # (batch_size, 32),
                            globals      =  {'node_padding_mask' : tf.TensorSpec(shape=(None, padding_n_node),    dtype=tf.bool),
                                             'edge_padding_mask' : tf.TensorSpec(shape=(None, padding_n_edge),    dtype=tf.bool)}, # (batch_size, 32), 
                            n_node       =   tf.TensorSpec(shape=(None, ),                  dtype=tf.int32), # (2*batch_size,), 
                            n_edge       =   tf.TensorSpec(shape=(None, ),                  dtype=tf.int32)) # (2*batch_size,)),
    elif n_partitions > 0:
        return jraph.GraphsTuple(nodes     =   tf.TensorSpec(shape=(n_partitions, None, num_node_features), dtype=tf.float32), # (32*batch_size, 8)
                            edges        =   tf.TensorSpec(shape=None                                                   ),    # No edges allowed
                            receivers    =   tf.TensorSpec(shape=(n_partitions, None, ),                  dtype=tf.int32),    # (64*batch_size,), 
                            senders      =   tf.TensorSpec(shape=(n_partitions, None, ),                  dtype=tf.int32), # (64*batch_size,), 
                            # globals      =   tf.TensorSpec(shape=(n_partitions, None, padding_n_node),    dtype=tf.bool), # (batch_size, 32), 
                            globals      =  {'node_padding_mask' : tf.TensorSpec(shape=(n_partitions, None, padding_n_node),    dtype=tf.bool),
                                             'edge_padding_mask' : tf.TensorSpec(shape=(n_partitions, None, padding_n_edge),    dtype=tf.bool)}, # (batch_size, 32),
                            n_node       =   tf.TensorSpec(shape=(n_partitions, None, ),                  dtype=tf.int32), # (2*batch_size,), 
                            n_edge       =   tf.TensorSpec(shape=(n_partitions, None, ),                  dtype=tf.int32)) # (2*batch_size,)),
