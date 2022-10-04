import os
import sys
import pickle
import time
import datetime
import numpy
import jax
from jax import numpy as jnp
import flax
from flax import serialization

from Rec2Odorant.odor_description.multiLabel.GNN.loader import Loader, Collate, DatasetBuilder
from Rec2Odorant.odor_description.multiLabel.GNN.MPNN.model.base_model import VanillaMPNN

from Rec2Odorant.odor_description.multiLabel.GNN.MPNN.make_init import make_init_model, get_tf_specs
from Rec2Odorant.odor_description.multiLabel.GNN.MPNN.make_create_optimizer import make_create_optimizer
from Rec2Odorant.odor_description.multiLabel.GNN.MPNN.make_predict import make_predict_epoch
import logging

def main_predict(hparams):    
    logdir = os.path.join(hparams['LOGGING_PARENT_DIR'])

    restore_file = hparams['RESTORE_FILE']

    model = VanillaMPNN(num_classes = hparams['NUM_CLASSES'], atom_features = hparams['ATOM_FEATURES'], bond_features = hparams['BOND_FEATURES'])

    logger = logging.getLogger('main_predict')
    logger.setLevel(logging.INFO)
    _datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join(logdir, model.__class__.__name__, _datetime)
    os.makedirs(logdir)
    # os.mkdir(os.path.join(logdir, 'ckpts'))
    logger_file_handler = logging.FileHandler(os.path.join(logdir, 'run.log'))
    logger_stdout_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(logger_file_handler)

    collate = Collate(mid_padding_n_node = hparams['PADDING_N_NODE'], mid_padding_n_edge = hparams['PADDING_N_EDGE'],
                    max_padding_n_node = hparams['PADDING_N_NODE'], max_padding_n_edge = hparams['PADDING_N_EDGE'], 
                    n_partitions = hparams['N_PARTITIONS'])

    predict_dataset = DatasetBuilder(data_csv = hparams['PREDICT_CSV_NAME'],
                        mol_col = hparams['MOL_COL'],
                        label_col = hparams['LABEL_COL'],
                        # weight_col = hparams['weight_col'],
                        atom_features = model.atom_features,# ['AtomicNum', 'ChiralTag', 'Hybridization', 'FormalCharge', 
                                # 'NumImplicitHs', 'ExplicitValence', 'Mass', 'IsAromatic'],
                        bond_features = model.bond_features, # ['BondType', 'IsAromatic'],
                        )

    _predict_loader = Loader(predict_dataset, 
                        batch_size = hparams['BATCH_SIZE'],
                        collate_fn = collate.make_collate(),
                        shuffle = False,
                        rng = jax.random.PRNGKey(int(time.time())),
                        drop_last = False,
                        n_partitions = hparams['N_PARTITIONS'])
        
    if hparams['LOADER_OUTPUT_TYPE'] == 'jax':
        predict_loader = _predict_loader
    elif hparams['LOADER_OUTPUT_TYPE'] == 'tf':
        predict_loader = _predict_loader.tf_Dataset(output_signature = get_tf_specs(is_weighted = False, prediction=True, n_node_features = len(hparams['ATOM_FEATURES']), 
                            n_edge_feature = len(hparams['BOND_FEATURES']), node_padding = hparams['PADDING_N_NODE'], edge_padding = hparams['PADDING_N_EDGE'], 
                            n_classes = hparams['NUM_CLASSES']))
        predict_loader = predict_loader.cache()
        predict_loader = predict_loader.shuffle(buffer_size = len(_predict_loader))
        predict_loader = predict_loader.prefetch(buffer_size = 4)
        logger.info('loader_output_type = {}'.format(hparams['LOADER_OUTPUT_TYPE']))

    key1, key2 = jax.random.split(jax.random.PRNGKey(int(time.time())), 2)
    key_params, _key_num_steps, key_num_steps, key_dropout = jax.random.split(key1, 4)

    # Initializations:
    start = time.time()
    logger.info('jax_version = {}'.format(jax.__version__))
    logger.info('flax_version = {}'.format(flax.__version__))
    logger.info('Initializing...')
    init_model = make_init_model(model, batch_size = hparams['BATCH_SIZE'], atom_features = model.atom_features, bond_features = model.bond_features)
    
    params = init_model(rngs = {'params' : key_params, 'dropout' : key_dropout})

    end = time.time()
    logger.info('TIME: init_model: {}'.format(end - start))

    create_optimizer = make_create_optimizer(model, option = hparams['OPTIMIZATION']['OPTION'], transition_steps = 800*(len(predict_dataset)/hparams['BATCH_SIZE']))
    init_state, scheduler = create_optimizer(params, rngs = {'dropout' : key_dropout}, learning_rate = hparams['LEARNING_RATE'])

    # Restore params:
    if restore_file is not None:
        logger.info('Restoring parameters from {}'.format(restore_file))
        with open(restore_file, 'rb') as pklfile:
            bytes_output = pickle.load(pklfile)
        state = serialization.from_bytes(init_state, bytes_output)
        logger.info('Parameters restored...')
    else:
        state = init_state    



    loss_option = hparams['LOSS_OPTION']
    predict_epoch = make_predict_epoch(num_classes = model.num_classes, loss_option = loss_option, logger = logger, return_embeddings=True, loader_output_type = hparams['LOADER_OUTPUT_TYPE'])

    # Combine predictions and data
    logits, embedding = predict_epoch(state, predict_loader)
    prediction = numpy.array(jax.nn.sigmoid(logits))
    embedding = numpy.array(embedding)

    df = predict_dataset.data.copy()
    df['prediction'] = list(prediction)
    df['embed'] = list(embedding)

    from shutil import copy
    export_dir = logdir
    copy(src = hparams['PREDICT_CSV_NAME'], dst = os.path.join(export_dir, 'pyrfume_embedding_2022_09_08_source.csv'))
    df[['_SMILES', 'prediction', 'embed']].to_json(os.path.join(export_dir, 'pyrfume_embedding_2022_09_08_embed.json'), orient = 'index')

    
