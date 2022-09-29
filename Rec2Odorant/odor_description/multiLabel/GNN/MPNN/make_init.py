import jax
from jax import numpy as jnp
import jraph
import time

import tensorflow as tf

from Rec2Odorant.odor_description.utils import create_line_graph_and_pad

def make_init_model(model, batch_size, atom_features, bond_features, n_partitions = 0):
    """
    Parameters:
    -----------
    seq_embedding_size : int
        size of embedding. 1024 for ProtBERT, 768 for TapeBERT
    """
    num_node_features = len(atom_features)
    num_edge_features = len(bond_features)
    key = jax.random.PRNGKey(int(time.time()))
    def init_model(rngs):

        padding_n_node = 32
        padding_n_edge = 64

        key_nodes, key_edges = jax.random.split(key)
        num_nodes = jax.random.randint(key_nodes, minval=10, maxval=padding_n_node - 1, shape = ())
        num_edges = jax.random.randint(key_edges, minval=30, maxval=padding_n_edge - 1, shape = ())



        _mol = jraph.GraphsTuple(nodes = jnp.ones(shape = (num_nodes, num_node_features), dtype = jnp.float32),
                                edges = jnp.ones(shape = (num_edges, num_edge_features), dtype = jnp.float32),
                                receivers = jnp.concatenate([jnp.arange(num_edges -1) + 1, jnp.array([0], dtype=jnp.int32)]),  
                                senders = jnp.arange(num_edges),                                                              
                                n_node = jnp.array([num_nodes]),
                                n_edge = jnp.array([num_edges]),
                                globals = None,
                                )
        batch = [_mol for _ in range(batch_size)]
        mols = []
        for mol in batch: 
            padded_mol = create_line_graph_and_pad(mol, 
                                                                    padding_n_node = padding_n_node, 
                                                                    padding_n_edge = padding_n_edge,
                                                                    )           
            
            
            
            
            mols.append(padded_mol)

        if n_partitions > 0:
            partition_size = len(batch) // n_partitions        
            mols = [jraph.batch(mols[i*partition_size:(i+1)*partition_size]) for i in range(n_partitions)]
        else:
            mols = jraph.batch(mols)
        params = model.init(rngs,  mols, deterministic = False, return_embeddings=False)
        return params
    return init_model


def get_tf_specs(n_partitions = 0, is_weighted = True, prediction = False, n_node_features = 8, n_edge_feature = 2, node_padding = 128, edge_padding = 265, n_classes = 195):
    """
    Get TensorSpec for tf.data.Dataset. These specs correspond to the output of the loader.
    """
    num_node_features = n_node_features
    num_edge_features = n_edge_feature

    padding_n_node = node_padding
    padding_n_edge = edge_padding

    G = jraph.GraphsTuple(nodes     =   tf.TensorSpec(shape=(None, num_node_features), dtype=tf.float32),
                        edges        =   tf.TensorSpec(shape=(None, num_edge_features), dtype=tf.float32),
                        receivers    =   tf.TensorSpec(shape=(None, ),                  dtype=tf.int32),    
                        senders      =   tf.TensorSpec(shape=(None, ),                  dtype=tf.int32), 
                        globals      =   tf.TensorSpec(shape=(None, padding_n_node),    dtype=tf.bool), 
                        n_node       =   tf.TensorSpec(shape=(None, ),                  dtype=tf.int32), 
                        n_edge       =   tf.TensorSpec(shape=(None, ),                  dtype=tf.int32)) 
    if prediction:
        if is_weighted:
            weights = (tf.TensorSpec(shape=(None, n_classes), dtype=tf.float32))
            return (G, weights)
        else:
            return (G,)
    else: 
        if is_weighted:
            labels =(tf.TensorSpec(shape=(None, n_classes), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, n_classes), dtype=tf.float32))
        else:
            labels = tf.TensorSpec(shape=(None, n_classes), dtype=tf.int32)
    return ( G, labels)