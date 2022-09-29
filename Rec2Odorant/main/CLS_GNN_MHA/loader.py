import re
import numpy
import pandas
import json
import jraph
import jax
from jax import numpy as jnp
import functools
import copy

from Rec2Odorant.mol2graph.utils import get_num_atoms_and_bonds

from Rec2Odorant.mol2graph.jraph.convert import smiles_to_jraph
from Rec2Odorant.mol2graph.exceptions import NoBondsError

from Rec2Odorant.main.utils import deserialize_to_jraph, create_line_graph_and_pad, create_line_graph, pad_graph_and_line_graph, pad_graph
from Rec2Odorant.main.base_loader import BaseDataset, BaseDataLoader


class ProtBERTDataset(BaseDataset):
    """
    consider introducing mol_buffer to save already preprocessed graphs.
    """
    def __init__(self, data_csv, seq_col, mol_col, label_col, weight_col = None,
                atom_features = ['AtomicNum'], bond_features = ['BondType'], 
                oversampling_function = None, class_weight_json = None, # TODO: Consider creating weight column with class weight inside directly.
                line_graph_max_size = None, # 10 * padding_n_node
                line_graph = True,
                # mid_padding_n_node = 100, mid_padding_n_edge = 220,
                # max_padding_n_node = 700, max_padding_n_edge = 1500,
                class_alpha = None,
                # orient='columns',
                **kwargs):
        """
        Parameters:
        -----------
        data_csv : str
            path to csv

        seq_col : str
            name of the column with protein sequence

        mol_col : str
            name of the column with smiles
        
        label_col : str
            name of the column with labels (in case of multilabel problem, 
            labels should be in one column)

        atom_features : list
            list of atom features.

        bond_features : list
            list of bond features.

        **kwargs
            IncludeHs
            sep
            seq_sep

        Notes:
        ------
        sequence representations are retrived in Collate.
        """
        self.data_csv = data_csv
        self.sep = kwargs.get('sep', ';')

        self.class_weight_json = class_weight_json

        self.mol_col = mol_col # 'SMILES'
        self.seq_col = seq_col # 'mutated_Sequence'
        self.label_col = label_col # 'Responsive'
        self.weight_col = weight_col

        self.class_alpha = class_alpha
        
        self.oversampling_function = oversampling_function

        self.IncludeHs = kwargs.get('IncludeHs', False)
        self.self_loops = kwargs.get('self_loops', False)

        self.atom_features = atom_features
        self.bond_features = bond_features

        self.line_graph = line_graph
        self.line_graph_max_size = line_graph_max_size # 10 * mid_padding_n_node

        self.data = self.read()

    def _read_graph(self, smiles):
        """
        """
        try:
            G = smiles_to_jraph(smiles, u = None, validate = False, IncludeHs = self.IncludeHs,
                            atom_features = self.atom_features, bond_features = self.bond_features,
                            self_loops = self.self_loops)
        except NoBondsError:
            return float('nan')
        # return G
        if self.line_graph:
            return (G, create_line_graph(G, max_size = self.line_graph_max_size))
        else:
            return (G, )

    def read(self):
        if isinstance(self.data_csv, pandas.DataFrame):
            if self.weight_col is not None:
                df = self.data_csv[[self.mol_col, self.seq_col, self.label_col, self.weight_col]]
            else:
                df = self.data_csv[[self.mol_col, self.seq_col, self.label_col]]
        else:
            if self.weight_col is not None:
                df = pandas.read_csv(self.data_csv, sep = self.sep, usecols = [self.mol_col, self.seq_col, self.label_col, self.weight_col])
            else:
                df = pandas.read_csv(self.data_csv, sep = self.sep, usecols = [self.mol_col, self.seq_col, self.label_col])
        
        if self.class_weight_json is not None:
            raise NotImplementedError('Check whether behaviour changed in CV. This will probably end up as a part of weight column.')
            with open(self.class_weight_json, 'r') as cls_dist_file:
                class_dist = json.load(cls_dist_file)
            if self.class_alpha is not None:
                class_weight_map = {0 : 1.0, 1 : (class_dist['0']/class_dist['1'])**self.class_alpha}
            else:
                class_weight_map = {0 : 1.0, 1 : (class_dist['0']/class_dist['1'])}
            class_weight_col = df[self.label_col].map(class_weight_map)
            if self.weight_col is not None:
                df[self.weight_col] = df[self.weight_col] * class_weight_col
            else:
                self.weight_col = 'sample_weight'
                df[self.weight_col] = class_weight_col

        smiles = df[self.mol_col].drop_duplicates()
        smiles.index = smiles
        smiles = smiles.apply(self._read_graph)
        smiles.dropna(inplace = True)
        smiles.rename('_graphs', inplace = True)

        df = df.join(smiles, on = self.mol_col, how = 'left')

        if self.oversampling_function is not None:
            df = self.oversampling_function(df, label_col = self.label_col)
        
        return df

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # TODO: previously jax DeviceArray and this raised assertionError 
        # in pandas. See if this behaviour changes in new padnas
        index = numpy.asarray(index)
        sample = self.data.iloc[index]
        
        seq = sample[self.seq_col]
        seq = ' '.join(list(seq))
        seq = re.sub(r"[UZOB]", "X", seq)

        mol = sample['_graphs']
        label = sample[self.label_col]
        if self.weight_col is not None:
            sample_weight = sample[self.weight_col]
            return seq, mol, (label, sample_weight)
        else:
            return seq, mol, label



def transpose_batch(batch):
    """
    Move first dimension of pytree into batch. I.e. pytree with (n_parts, n_elements, (batch_size, features, ...)) will be 
    changed to (n_elements, (n_parts, batch_size, features, ...)).
    
    Example:
    --------
    List of tuples [(X, Y, Z)] with dim(X) = (batch_size, x_size), dim(Y) = (batch_size, y_size), dim(Z) = (batch_size, z_size)
    is chaged to tuple (X', Y', Z') where dim(X') = (1, batch_size, x_size), dim(Y) = (1, batch_size, y_size), dim(Z) = (1, batch_size, z_size).
    """
    return jax.tree_multimap(lambda *x: jnp.stack(x, axis = 0), *batch)



class ProtBERTCollate:
    """
    Bookkeeping and clean manipulation with collate function.

    The aim of this class is to ease small modifications of some parts of collate function without
    the need for code redundancy. In Loader use output of make_collate as a collate function.
    """
    def __init__(self, tokenizer, padding_n_node, padding_n_edge, line_graph = True, n_partitions = 0, seq_max_length = 2048):
        self.tokenizer = tokenizer
        # self.mid_padding_n_node = mid_padding_n_node
        # self.mid_padding_n_edge = mid_padding_n_edge
        # self.max_padding_n_node = max_padding_n_node # max n_node in data: 421
        # self.max_padding_n_edge = max_padding_n_edge # max n_edge in data: 900
        self.padding_n_node = padding_n_node
        self.padding_n_edge = padding_n_edge
        self.n_partitions = n_partitions
        self.seq_max_length = seq_max_length

        if line_graph:
            self._graph_collate = functools.partial(self._graph_collate_with_line_graph, padding_n_node = padding_n_node, padding_n_edge = padding_n_edge, n_partitions = n_partitions)
        else:
            self._graph_collate = functools.partial(self._graph_collate_without_line_graph, padding_n_node = padding_n_node, padding_n_edge = padding_n_edge, n_partitions = n_partitions)

    def _seq_collate(self, batch):
        """
        """
        tokenizer = self.tokenizer
        n_partitions = self.n_partitions
        # seqs = dict(tokenizer(batch, return_tensors='np', padding = True))
        seqs = dict(tokenizer(batch, return_tensors='np', padding = 'max_length', max_length = self.seq_max_length, truncation = True)) # 2048
        if 'position_ids' not in seqs.keys():
                seqs['position_ids'] = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(seqs['input_ids']).shape[-1]), seqs['input_ids'].shape)
        if n_partitions > 0:
            partition_size = len(batch) // n_partitions
            _seqs = []
            for i in range(n_partitions): # n_partitions
                _seq = {}
                for key in seqs.keys():
                    _seq[key] = seqs[key][i*partition_size:(i+1)*partition_size]
                _seqs.append(_seq)
            return _seqs
        else:
            return seqs

    @staticmethod
    def _graph_collate_with_line_graph(batch, padding_n_node, padding_n_edge, n_partitions):
        """
        For n_edges in a padding graph for line graph see https://stackoverflow.com/questions/6548283/max-number-of-paths-of-length-2-in-a-graph-with-n-nodes
        We expect on average degree of a node to be 3 (C has max degree 4, but it has often implicit hydrogen/benzene ring/double bond)
        Error will be raised if the assumption is not enough.

        Notes:
        ------
        Most of the molecules have small number of edges, so for them padding can be small. Thus padding is branched into two branches, one for small graph
        and the other for big graphs. This will triger retracing twice in jitted processing, but for most molecules only small version will be used.
        """
        mols = []
        line_mols = []
        for mol, line_mol in batch:
            if len(mol.senders) == 0 or len(mol.receivers) == 0:
                print(mol)
                raise ValueError('Molecule with no bonds encountered.')
            if len(line_mol.senders) == 0 or len(line_mol.receivers) == 0:
                print(line_mol)
                raise ValueError('Molecule with no edges encountered (line molecule with no bonds).')
            padded_mol, padded_line_mol = pad_graph_and_line_graph(mol, 
                                                                line_mol, 
                                                                padding_n_node = padding_n_node, 
                                                                padding_n_edge = padding_n_edge)
            mols.append(padded_mol)
            line_mols.append(padded_line_mol)
        if n_partitions > 0:
            partition_size = len(batch) // n_partitions        
            return [(jraph.batch(mols[i*partition_size:(i+1)*partition_size]), 
                     jraph.batch(line_mols[i*partition_size:(i+1)*partition_size])) for i in range(n_partitions)]
        else:
            return (jraph.batch(mols), jraph.batch(line_mols))

    @staticmethod
    def _graph_collate_without_line_graph(batch, padding_n_node, padding_n_edge, n_partitions):
        """
        For n_edges in a padding graph for line graph see https://stackoverflow.com/questions/6548283/max-number-of-paths-of-length-2-in-a-graph-with-n-nodes
        We expect on average degree of a node to be 3 (C has max degree 4, but it has often implicit hydrogen/benzene ring/double bond)
        Error will be raised if the assumption is not enough.

        Notes:
        ------
        Most of the molecules have small number of edges, so for them padding can be small. Thus padding is branched into two branches, one for small graph
        and the other for big graphs. This will triger retracing twice in jitted processing, but for most molecules only small version will be used.
        """
        mols = []
        for mol in batch:
            mol = mol[0] # Output of dataset __getitem__ is a tuple even if line_graph = False.
            if len(mol.senders) == 0 or len(mol.receivers) == 0:
                print(mol)
                raise ValueError('Molecule with no bonds encountered.')
            padded_mol = pad_graph(mol, 
                                padding_n_node = padding_n_node, 
                                padding_n_edge = padding_n_edge)
            mols.append(padded_mol)
        if n_partitions > 0:
            partition_size = len(batch) // n_partitions        
            return [jraph.batch(mols[i*partition_size:(i+1)*partition_size]) for i in range(n_partitions)]
        else:
            return jraph.batch(mols)

    def _label_collate(self, batch):
        """
        """
        n_partitions = self.n_partitions
        if n_partitions > 0:
            partition_size = len(batch) // n_partitions
            return [jnp.stack(batch[i*partition_size:(i+1)*partition_size]) for i in range(n_partitions)] # [numpy.stack(batch[i*partition_size:(i+1)*partition_size]) for i in range(n_partitions)]
        else:
            return jnp.stack(batch) # numpy.stack

    def make_collate(self):
        """
        Create collate function that is the input to Loader.
        """
        n_partitions = self.n_partitions
        
        def _collate(batch):
            if isinstance(batch[0], str):
                return self._seq_collate(batch)
            # elif isinstance(batch[0], jraph.GraphsTuple): # TODO: Beware here because of [0][0]. 
            #     return self._graph_collate(batch)
            elif isinstance(batch[0], numpy.integer) or isinstance(batch[0], numpy.floating):
                return self._label_collate(batch)
            elif isinstance(batch[0], (tuple,list)): # <------ DEBUG
                if isinstance(batch[0][0], jraph.GraphsTuple):
                    return self._graph_collate(batch)
                else:
                    transposed = zip(*batch)
                    _batch = tuple([_collate(samples) for samples in transposed]) # [_collate(samples) for samples in transposed]
                    if n_partitions > 0:
                        return tuple(zip(*_batch)) # list(zip(*_batch))
                    else:
                        return _batch
            else:
                raise ValueError('Unexpected type passed from dataset to loader: {}'.format(type(batch[0])))

        def collate(batch):
            batch = _collate(batch)
            if n_partitions > 0:
                batch = transpose_batch(batch)
            return batch
        
        return collate


class ProtBERTLoader(BaseDataLoader):
    """
    Paramters:
    ----------
    padding_n_node : int
        maximum number of nodes in one graph. Final padding size of a batch will be padding_n_nodes * batch_size

    padding_n_edge : int
        maximum number of edges in one graph. Final padding size of a batch will be padding_n_nodes * batch_size
    """
    def __init__(self, dataset, collate_fn,
                    batch_size=1,
                    n_partitions = 0,
                    shuffle=False, 
                    rng=None, 
                    drop_last=False):

        self.n_partitions = n_partitions
        if n_partitions > 0:
            assert batch_size % self.n_partitions == 0

        super(self.__class__, self).__init__(dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        rng = rng,
        drop_last = drop_last,
        collate_fn = collate_fn,
        )


# --------------------
# Precomputed ProtBERT
# --------------------
class ProtBERTDatasetPrecomputeBERT(BaseDataset):
    """
    consider introducing mol_buffer to save already preprocessed graphs.
    """
    def __init__(self, data_csv, seq_id_col, mol_col, label_col, weight_col = None,
                atom_features = ['AtomicNum'], bond_features = ['BondType'], 
                oversampling_function = None, class_weight_json = None, # TODO: Consider creating weight column with class weight inside directly.
                line_graph_max_size = None, # 10 * padding_n_node
                line_graph = True,
                # mid_padding_n_node = 100, mid_padding_n_edge = 220,
                # max_padding_n_node = 700, max_padding_n_edge = 1500,
                class_alpha = None,
                # orient='columns',
                **kwargs):
        """
        Parameters:
        -----------
        data_csv : str
            path to csv

        seq_id_col : str
            name of the column with protein IDs

        mol_col : str
            name of the column with smiles
        
        label_col : str
            name of the column with labels (in case of multilabel problem, 
            labels should be in one column)

        atom_features : list
            list of atom features.

        bond_features : list
            list of bond features.

        **kwargs
            IncludeHs
            sep
            seq_sep

        Notes:
        ------
        sequence representations are retrived in Collate.
        """
        self.data_csv = data_csv
        self.sep = kwargs.get('sep', ';')

        self.class_weight_json = class_weight_json

        # self.seq_json = seq_json
        # self.poc_csv = poc_csv

        self.mol_col = mol_col # 'SMILES'
        self.seq_id_col = seq_id_col # 'Gene'
        self.label_col = label_col # 'Responsive'
        self.weight_col = weight_col

        self.class_alpha = class_alpha
        
        self.oversampling_function = oversampling_function

        self.IncludeHs = kwargs.get('IncludeHs', False)
        self.self_loops = kwargs.get('self_loops', False)

        self.atom_features = atom_features
        self.bond_features = bond_features

        self.line_graph = line_graph
        self.line_graph_max_size = line_graph_max_size # 10 * mid_padding_n_node

        # self.df_poc_cols_ordered = ['pock_volume', 'pock_asa', 'pock_pol_asa', 'pock_apol_asa',
        #                             'pock_asa22', 'pock_pol_asa22', 'pock_apol_asa22', 'nb_AS',
        #                             'mean_as_ray', 'mean_as_solv_acc', 'apol_as_prop', 'mean_loc_hyd_dens',
        #                             'hydrophobicity_score', 'volume_score', 'polarity_score',
        #                             'charge_score', 'prop_polar_atm', 'as_density', 'as_max_dst',
        #                             'convex_hull_volume', 'nb_abpa', 'ALA', 'CYS', 'ASP', 'GLU', 'PHE',
        #                             'GLY', 'HIS', 'ILE', 'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG',
        #                             'SER', 'THR', 'VAL', 'TRP', 'TYR']
        self.data = self.read()
        # self.df = self.data.copy()

    def _read_graph(self, smiles):
        """
        """
        try:
            G = smiles_to_jraph(smiles, u = None, validate = False, IncludeHs = self.IncludeHs,
                            atom_features = self.atom_features, bond_features = self.bond_features,
                            self_loops = self.self_loops)
        except NoBondsError:
            return float('nan')
        # return G
        if self.line_graph:
            return (G, create_line_graph(G, max_size = self.line_graph_max_size))
        else:
            return (G, )

    def read(self):
        if isinstance(self.data_csv, pandas.DataFrame):
            if self.weight_col is not None:
                df = self.data_csv[[self.mol_col, self.seq_id_col, self.label_col, self.weight_col]]
            else:
                df = self.data_csv[[self.mol_col, self.seq_id_col, self.label_col]]
        else:
            if self.weight_col is not None:
                df = pandas.read_csv(self.data_csv, sep = self.sep, usecols = [self.mol_col, self.seq_id_col, self.label_col, self.weight_col])
            else:
                df = pandas.read_csv(self.data_csv, sep = self.sep, usecols = [self.mol_col, self.seq_id_col, self.label_col])
        
        if self.class_weight_json is not None:
            raise NotImplementedError('Check whether behaviour changed in CV. This will probably end up as a part of weight column.')
            with open(self.class_weight_json, 'r') as cls_dist_file:
                class_dist = json.load(cls_dist_file)
            if self.class_alpha is not None:
                class_weight_map = {0 : 1.0, 1 : (class_dist['0']/class_dist['1'])**self.class_alpha}
            else:
                class_weight_map = {0 : 1.0, 1 : (class_dist['0']/class_dist['1'])}
            class_weight_col = df[self.label_col].map(class_weight_map)
            if self.weight_col is not None:
                df[self.weight_col] = df[self.weight_col] * class_weight_col
            else:
                self.weight_col = 'sample_weight'
                df[self.weight_col] = class_weight_col

        smiles = df[self.mol_col].drop_duplicates()
        smiles.index = smiles
        smiles = smiles.apply(self._read_graph)
        smiles.dropna(inplace = True)
        smiles.rename('_graphs', inplace = True)

        df = df.join(smiles, on = self.mol_col, how = 'left')

        if self.oversampling_function is not None:
            df = self.oversampling_function(df, label_col = self.label_col)
        
        return df

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # TODO: previously jax DeviceArray and this raised assertionError 
        # in pandas. See if this behaviour changes in new padnas
        index = numpy.asarray(index)
        sample = self.data.iloc[index]
        
        seq_id = sample[self.seq_id_col]
        seq_id = str(seq_id)
        
        mol = sample['_graphs']
        # if mol[0].n_node >= 32 or mol[0].n_edge >= 64:
        #     print(sample[self.mol_col])
        #     print(get_num_atoms_and_bonds(sample[self.mol_col]))
        #     print(mol[0])
        #     raise Exception('FUCK')
        
        # mol = deserialize_to_jraph(mol)

        if mol is None:
            print(sample)
            print(mol)
            raise Exception('KURWAAAA')

        if not mol == mol:
            print(sample)
            print(mol)
            raise Exception('KURWAAAA')

        label = sample[self.label_col]
        if self.weight_col is not None:
            sample_weight = sample[self.weight_col]
            return seq_id, mol, (label, sample_weight)
        else:
            return seq_id, mol, label



# ------------------------
# Precomputed ProtBERT CLS
# ------------------------
class ProtBERTCollatePrecomputeBERT_CLS(ProtBERTCollate):
    def __init__(self, bert_table, padding_n_node, padding_n_edge, n_partitions, line_graph = True, from_disk = False):
        self.bert_table = bert_table
        # self.mid_padding_n_node = mid_padding_n_node
        # self.mid_padding_n_edge = mid_padding_n_edge
        # self.max_padding_n_node = max_padding_n_node # max n_node in data: 421
        # self.max_padding_n_edge = max_padding_n_edge # max n_edge in data: 900
        self.padding_n_node = padding_n_node
        self.padding_n_edge = padding_n_edge
        self.n_partitions = n_partitions

        if line_graph:
            self._graph_collate = functools.partial(self._graph_collate_with_line_graph, padding_n_node = padding_n_node, padding_n_edge = padding_n_edge, n_partitions = n_partitions)
        else:
            self._graph_collate = functools.partial(self._graph_collate_without_line_graph, padding_n_node = padding_n_node, padding_n_edge = padding_n_edge, n_partitions = n_partitions)

        if not from_disk:
            bert_dict = {}
            # print(self.bert_table.read())
            for row in self.bert_table.iterrows():
                bert_dict[row['id'].decode('utf-8')] = row['hidden_states']
            print('Table loaded...')
            self._seq_collate = functools.partial(self._seq_collate_from_ram, bert_dict = bert_dict, n_partitions = n_partitions)
        else:
            self._seq_collate = functools.partial(self._seq_collate_from_disk, bert_table = bert_table, n_partitions = n_partitions)
        
    @staticmethod
    def _seq_collate_from_disk(batch, bert_table, n_partitions):
        # bert_table = self.bert_table
        # n_partitions = self.n_partitions
        seqs_hidden_states = []
        for _id in batch:
            x = list(bert_table.where('(id == b"{}")'.format(_id)))
            if len(x) == 0:
                raise ValueError('No record found in bert_table for id: {}'.format(_id))
            seqs_hidden_states.append(x[0]['hidden_states'])

        if n_partitions > 0:
            seqs = []
            partition_size = len(batch) // n_partitions
            for i in range(n_partitions):
                # seqs.append({'hidden_states' : numpy.stack(seqs_hidden_states[i*partition_size:(i+1)*partition_size])})
                # seqs.append(numpy.stack(seqs_hidden_states[i*partition_size:(i+1)*partition_size]))
                seqs.append(jnp.stack(seqs_hidden_states[i*partition_size:(i+1)*partition_size]))
        else:
            # seqs = {'hidden_states' : numpy.stack(seqs_hidden_states)}
            # seqs = numpy.stack(seqs_hidden_states)
            seqs = jnp.stack(seqs_hidden_states)
        return seqs

    @staticmethod
    def _seq_collate_from_ram(batch, bert_dict, n_partitions):
        # bert_table = self.bert_table
        # n_partitions = self.n_partitions
        seqs_hidden_states = []
        for _id in batch:
            x = bert_dict[_id]
            seqs_hidden_states.append(x)

        if n_partitions > 0:
            seqs = []
            partition_size = len(batch) // n_partitions
            for i in range(n_partitions):
                # seqs.append(numpy.stack(seqs_hidden_states[i*partition_size:(i+1)*partition_size]))
                seqs.append(jnp.stack(seqs_hidden_states[i*partition_size:(i+1)*partition_size]))
                # seqs.append({'hidden_states' : numpy.stack(seqs_hidden_states[i*partition_size:(i+1)*partition_size])})
        else:
            # seqs = numpy.stack(seqs_hidden_states)
            seqs = jnp.stack(seqs_hidden_states)
            # seqs = {'hidden_states' : numpy.stack(seqs_hidden_states)}

        return seqs




if __name__ == '__main__':
    import os
    from transformers import BertTokenizer
    from transformers import FlaxBertModel, BertConfig
    import time
    import flax

    from Receptor_odorant.JAX.BERT_GNN.CLS_GTransformer.make_init import get_tf_specs
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    if True:
        hparams = {'BATCH_SIZE' : 800, # 30,
                    'N_EPOCH' : 10000,
                    'N_PARTITIONS' : 0,
                    'LEARNING_RATE' : 0.001,
                    'SAVE_FREQUENCY' : 50,
                    'LOG_IMAGES_FREQUENCY' : 50,
                    'LOSS_OPTION' : 'cross_entropy',
                    'CLASS_ALPHA' : None, # 1.5
                    'LINE_GRAPH_MAX_SIZE_MULTIPLIER' : 5,
                    # model hparmas:
                    'ATOM_FEATURES' : ('AtomicNum', 'ChiralTag', 'Hybridization', 'FormalCharge', 
                                    'NumImplicitHs', 'ExplicitValence', 'Mass', 'IsAromatic'), # None,
                    'BOND_FEATURES' : ('BondType', 'Stereo', 'IsAromatic'), # None,
                    }
        
        _datacase = os.path.join('chemosimdb', 'all_20220513-203629', 'EC50_random_data',  '20220513-203728', 'quality__screening_weight', 'mix__concatGraph')
        _h5file = os.path.join('BERT_GNN', 'Data', 'chemosimdb', 'all_20220513-203629', 'PrecomputeProtBERT_CLS', 'ProtBERT_CLS.h5')
        datadir = os.path.join('BERT_GNN',  'Data', _datacase)

        dataparams = {'DATACASE' : _datacase,
                    'SIZE_CUT' : 'size_cut_atom32_bond64', # '' if None
                    'TRAIN' : 'data_train_small.csv',
                    'VALID' : 'data_valid_small.csv',
                    'BERT_H5FILE' : _h5file,
                    'seq_id_col' : 'seq_id', # 'Gene',
                    'mol_col' : '_SMILES', # 'SMILES',
                    'label_col' : 'Responsive',
                    'weight_col' : None, # 'sample_weight', # 'sample_weight',
                    'valid_weight_col' : None,
                    'loader_output_type' : 'tf',
                    }

        import tables
        h5file = tables.open_file(dataparams['BERT_H5FILE'], mode = 'r', title="TapeBERT")
        bert_table = h5file.root.bert.BERTtable
        from_disk = False

        collate = ProtBERTCollatePrecomputeBERT_CLS(bert_table, 
                                                    padding_n_node = 32, padding_n_edge = 64,
                                                    n_partitions = hparams['N_PARTITIONS'],
                                                    from_disk = from_disk)

        dataset = ProtBERTDatasetPrecomputeBERT(data_csv = os.path.join(datadir, dataparams['SIZE_CUT'], dataparams['TRAIN']),
                            class_weight_json = None, # os.path.join(datadir, 'class_dist.json'),
                            mol_col = dataparams['mol_col'],
                            seq_id_col = dataparams['seq_id_col'], # Gene is only sequence id.
                            label_col = dataparams['label_col'],
                            weight_col = dataparams['weight_col'],
                            atom_features = hparams['ATOM_FEATURES'],# ['AtomicNum', 'ChiralTag', 'Hybridization', 'FormalCharge', 
                                    # 'NumImplicitHs', 'ExplicitValence', 'Mass', 'IsAromatic'],
                            bond_features = hparams['BOND_FEATURES'], # ['BondType', 'IsAromatic'],
                            class_alpha = None,
                            line_graph_max_size = hparams['LINE_GRAPH_MAX_SIZE_MULTIPLIER'] * collate.padding_n_node,
                            )

        _loader = ProtBERTLoader(dataset, 
                            batch_size = hparams['BATCH_SIZE'],
                            collate_fn = collate.make_collate(),
                            shuffle = False,     # NOTE: shuffle is redundant for tf.data.Dataset here.
                            rng = jax.random.PRNGKey(int(time.time())),
                            drop_last = False,
                            n_partitions = hparams['N_PARTITIONS'])

        valid_dataset = ProtBERTDatasetPrecomputeBERT(data_csv = os.path.join(datadir, dataparams['SIZE_CUT'], dataparams['VALID']),
                            mol_col = dataparams['mol_col'],
                            seq_id_col = dataparams['seq_id_col'], # Gene is only sequence id.
                            label_col = dataparams['label_col'],
                            weight_col = dataparams['valid_weight_col'],
                            atom_features = hparams['ATOM_FEATURES'], # ['AtomicNum', 'ChiralTag', 'Hybridization', 'FormalCharge', 
                                    # 'NumImplicitHs', 'ExplicitValence', 'Mass', 'IsAromatic'],
                            bond_features = hparams['BOND_FEATURES'], # ['BondType', 'IsAromatic'],
                            line_graph_max_size = hparams['LINE_GRAPH_MAX_SIZE_MULTIPLIER'] * collate.padding_n_node,
                            )

        _valid_loader = ProtBERTLoader(valid_dataset, 
                        batch_size = hparams['BATCH_SIZE'],
                        collate_fn = collate.make_collate(),
                        shuffle = False,  # NOTE: shuffle is redundant for tf.data.Dataset here.
                        rng = jax.random.PRNGKey(int(time.time())),
                        drop_last = False,
                        n_partitions = hparams['N_PARTITIONS'])

        # print(dataset.data[['seq_id', '_SMILES', 'Responsive']])
        # print(valid_dataset.data[['seq_id', '_SMILES', 'Responsive']])
        big_dataset = dataset + valid_dataset
        _big_loader = ProtBERTLoader(big_dataset, 
                        batch_size = hparams['BATCH_SIZE'],
                        collate_fn = collate.make_collate(),
                        shuffle = False,  # NOTE: shuffle is redundant for tf.data.Dataset here.
                        rng = jax.random.PRNGKey(int(time.time())),
                        drop_last = False,
                        n_partitions = hparams['N_PARTITIONS'])

        for i, ele in enumerate(_big_loader):
            print(i*hparams['BATCH_SIZE'])
            


    if False:
        tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )


        dataset = ProtBERTDatasetPrecomputeBERT(data_csv = os.path.join('..', 'BERT_GNN', 
                                                        'Data', 
                                                        'ECRO2021_CV',
                                                        'bOR_bODOR_seq_BERT_CLS',
                                                        'random_data', 
                                                        '20210902-110813',
                                                        'weighted_LogHarmonic',
                                                        'data_train.csv'),
                        mol_col = 'SMILES',
                        seq_id_col = 'Gene', # Gene is only sequence id.
                        label_col = 'Responsive',
                        weight_col = 'sample_weight',
                        )

        import tables
        h5file = tables.open_file(os.path.join('BERT_GNN', 'Data', 'PrecomputeProtBERT_CLS','Sequence_unaligned_ProtBERT_CLS.h5'), mode = 'r', title="ProtBERT")
        bert_table = h5file.root.bert.BERTtable
        collate = ProtBERTCollatePrecomputeBERT_CLS(bert_table, 
                                                padding_n_node = 32, padding_n_edge = 64,
                                                # max_padding_n_node = 128, max_padding_n_edge = 256, 
                                                n_partitions = 0,
                                                from_disk = False)

        loader = ProtBERTLoader(dataset, 
                                batch_size = 800, 
                                collate_fn = collate.make_collate(),
                                shuffle = False,
                                rng = jax.random.PRNGKey(int(time.time())),
                                drop_last = False,
                                n_partitions = 0)


        import collections
        import itertools
        from jax.interpreters import xla
        from flax.jax_utils import _pmap_device_order
        def prefetch_to_device(iterator, size, devices=None):
            queue = collections.deque()
            devices = devices or _pmap_device_order()

            def _prefetch(xs):
              if hasattr(jax, "device_put_sharded"):  # jax>=0.2.0
                return jax.device_put_sharded(list(xs), devices)
              else:
                aval = jax.xla.abstractify(xs)
                assert xs.shape[0] == len(devices), (
                    "The first dimension of the iterator's ndarrays is not "
                    "equal to the number of devices.")
                buffers = [xla.device_put(x, devices[i])
                           for i, x in enumerate(xs)]
                return jax.pxla.ShardedDeviceArray(aval, buffers)

            def enqueue(n):  # Enqueues *up to* `n` elements from the iterator.
              for data in itertools.islice(iterator, n):
                queue.append(jax.tree_map(_prefetch, data))

            enqueue(size)  # Fill up the buffer.
            while queue:
              yield queue.popleft()
              enqueue(1)

        loader = prefetch_to_device(loader)

        times = []
        for _ in range(5):
            print('Begin...')
            # _loader = loader
            # _loader = iter(loader)
            # _loader = flax.jax_utils.prefetch_to_device(_loader, size = 2)
            start = time.time()
            for i, batch in enumerate(loader):
                # print(jax.tree_map(lambda x: x.shape, batch))
                print(i)
                end = time.time()
                print(list(batch[0].keys()))
                print(batch[0]['hidden_states'].shape)
                # print(batch[0]['attention_mask'].shape)
                _time = end - start
                times.append(_time)
                print('time: {}'.format(_time))
                print('---------')
                print(jax.tree_map(lambda x: x.shape, batch))
                start = end
            loader.reset()

        from matplotlib import pyplot as plt
        _bla = pandas.Series(times)
        _bla.hist(bins = 100)
        plt.show()

        h5file.close()