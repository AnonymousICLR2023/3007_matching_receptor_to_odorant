# -------------------------------------------------------------------------------------
# NOTE: All postprocess functions are not changing data_test (except CVPP for mixtures)
# -------------------------------------------------------------------------------------
import json
import os
from shutil import copy

import numpy
import pandas
from matplotlib import pyplot as plt
from Rec2Odorant.mol2graph.utils import get_num_atoms_and_bonds
from Rec2Odorant.main.base_cross_validation import BaseCVPostProcess


# --------
# Weights:
# --------
class CVPP_addWeights_Harmonic(BaseCVPostProcess):
    """
    Notes:
    ------
    This weight is depandant on split.
    """
    def __init__(self, data_dir, k = 100.0):
        name = None # 'weighted_Harmonic'
        super(CVPP_addWeights_Harmonic, self).__init__(name, data_dir)
        self.k = k
        self.col_name = 'harmonic'

    def serialize_hparams(self):
        """
        returns dictionary with all hyperparameters that will be saved. self.working_dir will be added
        to the dict in self.save_hparams.
        """
        return {'func_weight_name' : 'reciprocal_harmonic_mean_k100',
                'k' : self.k}

    def _weight_function(self, x):
        """
        change k for scaling and numerical stability.
        """
        return 0.5*(self.k/float(x['count_Gene']) + self.k/float(x['count_CID']))

    def _postprocess(self, data):
        # raise NotImplementedError('Behaviour change: weights must be added to separate columns in the data.')
        count_gene = data.groupby(by = 'seq_id').count()['Responsive']
        count_gene.name = 'count_Gene'        
        count_CID = data.groupby(by = 'mol_id').count()['Responsive']
        count_CID.name = 'count_CID'
        data = data.join(count_gene, on = 'seq_id', how = 'inner')
        data = data.join(count_CID, on = 'mol_id', how = 'inner')
        data[self.col_name + '_weight'] = data.apply(self._weight_function, axis = 1)
        data.drop('count_CID', axis=1, inplace = True)
        data.drop('count_Gene', axis=1, inplace = True)
        return data

    def postprocess(self):
        data_train, data_valid, data_test = self.load_data()

        if not data_train.empty:
            data_train = self._postprocess(data_train)
        data_train.to_csv(os.path.join(self.working_dir, 'data_train.csv'), sep=';', index = False, header = True)
        if not data_valid.empty:
            data_valid = self._postprocess(data_valid)
        data_valid.to_csv(os.path.join(self.working_dir, 'data_valid.csv'), sep=';', index = False, header = True)
        # if not data_test.empty:
        #     data_test = self._postprocess(data_test)
        data_test.to_csv(os.path.join(self.working_dir, 'data_test.csv'), sep=';', index = False, header = True)

        # copy(src = os.path.join(self.data_dir, 'mols.csv'),
        #     dst = os.path.join(self.working_dir, 'mols.csv'))
        # copy(src = os.path.join(self.data_dir, 'seqs.csv'),
        #     dst = os.path.join(self.working_dir, 'seqs.csv'))

        self.save_hparams(prefix = self.col_name + '_')
        return None
        

class CVPP_addWeights_LogHarmonic(CVPP_addWeights_Harmonic):
    """
    Notes:
    ------
    This weight is depandant on split.
    """
    def __init__(self, data_dir, k = 100.0):
        name = None # 'weighted_LogHarmonic'
        super(CVPP_addWeights_Harmonic, self).__init__(name, data_dir)
        self.k = k
        self.col_name = 'logHarmonic'

    def _weight_function(self, x):
        """
        change k for scaling and numerical stability.
        """
        return numpy.log(0.5*(self.k/float(x['count_Gene']) + self.k/float(x['count_CID'])) + 1)


class CVPP_class_dist(BaseCVPostProcess):
    """
    Notes:
    ------
    This weight is depandant on split.
    """
    def __init__(self, data_dir):
        name = None
        super(CVPP_class_dist, self).__init__(name, data_dir)

    def serialize_hparams(self):
        """
        returns dictionary with all hyperparameters that will be saved. self.working_dir will be added
        to the dict in self.save_hparams.
        """
        return {}

    def postprocess(self):
        data_train, data_valid, _ = self.load_data()

        data = []
        if not data_train.empty:
            data.append(data_train)
        if not data_valid.empty:
            data.append(data_valid)

        data = pandas.concat(data)

        cls_dist = data.groupby('Responsive').count()['seq_id'].to_dict()

        with open(os.path.join(self.working_dir, 'class_dist.json'), 'w') as outfile:
            json.dump(cls_dist, outfile)

        # self.save_hparams()
        return None


class CVPP_addWeights_Class(BaseCVPostProcess):
    """
    Notes:
    ------
    This weight is depandant on split.
    """
    def __init__(self, data_dir, auxiliary_data_path):
        name = None # 'weights__LogHarmonic' + '_' + 'class'
        super(CVPP_addWeights_Class, self).__init__(name, data_dir)

        self.auxiliary_data_path = auxiliary_data_path

    def serialize_hparams(self):
        """
        returns dictionary with all hyperparameters that will be saved. self.working_dir will be added
        to the dict in self.save_hparams.
        """
        return {'data_dir' : self.data_dir,
                'auxiliary_data_path' : self.auxiliary_data_path}

    def load_auxiliary(self):
        auxiliary = {}
        with open(self.auxiliary_data_path['class_dist'], 'r') as jsonfile:
            auxiliary['class_dist'] = json.load(jsonfile)

        with open(os.path.join(self.working_dir, 'addWeights_Class_auxiliary.json'), 'w') as jsonfile:
            json.dump(auxiliary, jsonfile)

        return auxiliary

    def _postprocess(self, data, auxilary):
        class_weight_map = {0 : 1.0, 1 : (auxilary['class_dist']['0']/auxilary['class_dist']['1'])}
        # class_weight_map = {0 : 1.0 + (auxilary['class_dist']['1']/auxilary['class_dist']['0']), 1 : 1.0 + (auxilary['class_dist']['0']/auxilary['class_dist']['1'])}
        data['class_weight'] = data['Responsive'].map(class_weight_map)
        # data['sample_weight'] = data['LogHarmonic_weight'] * data['class_weight']
        return data

    def postprocess(self):
        data_train, data_valid, data_test = self.load_data()
        auxilary = self.load_auxiliary()

        if not data_train.empty:
            data_train = self._postprocess(data_train, auxilary)
        data_train.to_csv(os.path.join(self.working_dir, 'data_train.csv'), sep=';', index = False, header = True)
        if not data_valid.empty:
            data_valid = self._postprocess(data_valid, auxilary)
        data_valid.to_csv(os.path.join(self.working_dir, 'data_valid.csv'), sep=';', index = False, header = True)
        # if not data_test.empty:
        #     data_test = self._postprocess(data_test, auxilary)
        data_test.to_csv(os.path.join(self.working_dir, 'data_test.csv'), sep=';', index = False, header = True)

        self.save_hparams(prefix = 'addWeights_Class_')
        return None


# ---------------
# Combine Weights
# ---------------
class CVPP_combineWeights(BaseCVPostProcess):
    """
    Combine weight columns by multiplication.
    """
    def __init__(self, data_dir, weight_cols):
        name = None # 'weights__LogHarmonic' + '_' + 'class'
        super(CVPP_combineWeights, self).__init__(name, data_dir)

        self.weight_cols = weight_cols

    def serialize_hparams(self):
        """
        returns dictionary with all hyperparameters that will be saved. self.working_dir will be added
        to the dict in self.save_hparams.
        """
        return {'data_dir' : self.data_dir,
                'weight_cols' : self.weight_cols}

    def _postprocess(self, data):
        data['sample_weight'] = data[self.weight_cols].prod(axis = 1)
        return data

    def postprocess(self):
        data_train, data_valid, data_test = self.load_data()

        if not data_train.empty:
            data_train = self._postprocess(data_train)
        data_train.to_csv(os.path.join(self.working_dir, 'data_train.csv'), sep=';', index = False, header = True)
        if not data_valid.empty:
            data_valid = self._postprocess(data_valid)
        data_valid.to_csv(os.path.join(self.working_dir, 'data_valid.csv'), sep=';', index = False, header = True)
        # if not data_test.empty:
        #     data_test = self._postprocess(data_test)
        data_test.to_csv(os.path.join(self.working_dir, 'data_test.csv'), sep=';', index = False, header = True)

        self.save_hparams(prefix = 'combineWeights_')
        return None
    

# --------
# Size cut
# --------
# class CVPP_SizeCut(BaseCVPostProcess):
#     def __init__(self, data_dir, n_atom_threshold = 32, n_bond_threshold = 64, mol_id_col = 'mol_id', mol_col = '_SMILES', bond_multiplier = 2):
#         """
#         Notes:
#         ------
#         n_atom_threshold = 32, n_bond_threshold = 64 was chosen so that we don't loose too much (around ~43 unique molecules/mixtures, ~4000 pairs) but the graphs are small enough.
# 
#         data with num_atoms==threshold is included in data_big. 
#         """
#         name = 'size_cut_' + 'atom' + str(n_atom_threshold) + '_' + 'bond' + str(n_bond_threshold)
#         super(CVPP_SizeCut, self).__init__(name, data_dir)
#         self.n_atom_threshold = n_atom_threshold
#         self.n_bond_threshold = n_bond_threshold
#         self.mol_id_col = mol_id_col
#         self.mol_col = mol_col
#         self.bond_multiplier = bond_multiplier # Because of directed edges.
# 
#     def serialize_hparams(self):
#         """
#         returns dictionary with all hyperparameters that will be saved. self.working_dir will be added
#         to the dict in self.save_hparams.
#         """
#         return {'n_atom_threshold' : self.n_atom_threshold,
#                 'n_bond_threshold' : self.n_bond_threshold}
#     
#     def _test_big(self, x):
#         num_atoms, num_bonds = get_num_atoms_and_bonds(x)        
#         return num_atoms >= self.n_atom_threshold or self.bond_multiplier * num_bonds >= self.n_bond_threshold
# 
#     def _postprocess(self, data):
#         """
#         """
#         _data = data.groupby(self.mol_id_col).first()[self.mol_col].copy()
#         big_idx = _data[_data.apply(lambda x: self._test_big(x))].index
#         
#         data_small = data[~(data[self.mol_id_col].isin(big_idx))]
#         data_big   = data[ (data[self.mol_id_col].isin(big_idx))]
# 
#         print('Num of unique molecules in big: {}'.format(len(big_idx)))
#         print('Num of pairs in big: {}'.format(len(data_big)))
# 
#         return data_small, data_big
# 
#     def postprocess(self):
#         data_train, data_valid, data_test = self.load_data()
# 
#         if not data_train.empty:
#             data_train_small, data_train_big = self._postprocess(data_train)
#         else:
#             data_valid_small = data_train.copy()
#             data_train_big = data_train.copy()
#         data_train_small.to_csv(os.path.join(self.working_dir, 'data_train_small.csv'), sep=';', index = False, header = True)
#         data_train_big.to_csv(os.path.join(self.working_dir, 'data_train_big.csv'), sep=';', index = False, header = True)
# 
#         if not data_valid.empty:
#             data_valid_small, data_valid_big = self._postprocess(data_valid)
#         else:
#             data_valid_small = data_valid.copy()
#             data_valid_big = data_valid.copy()
#         data_valid_small.to_csv(os.path.join(self.working_dir, 'data_valid_small.csv'), sep=';', index = False, header = True)
#         data_valid_big.to_csv(os.path.join(self.working_dir, 'data_valid_big.csv'), sep=';', index = False, header = True)
# 
#         if not data_test.empty:
#             data_test_small, data_test_big = self._postprocess(data_test)
#         else:
#             data_test_small = data_test.copy()
#             data_test_big = data_test.copy()
#         data_test_small.to_csv(os.path.join(self.working_dir, 'data_test_small.csv'), sep=';', index = False, header = True)
#         data_test_big.to_csv(os.path.join(self.working_dir, 'data_test_big.csv'), sep=';', index = False, header = True)
# 
#         self.save_hparams()
#         return None


class CVPP_SizeCut(BaseCVPostProcess):
    def __init__(self, data_dir, n_node_thresholds = [32], n_edge_thresholds = [64], mol_id_col = 'mol_id', mol_col = '_SMILES', bond_multiplier = 2):
        """
        Notes:
        ------
        n_node_threshold = 32, n_edge_threshold = 64 was chosen so that we don't loose too much (around ~43 unique molecules/mixtures, ~4000 pairs) but the graphs are small enough.

        data with num_atoms==threshold is included in data_big. 
        """
        name = 'size_cut'
        super(CVPP_SizeCut, self).__init__(name, data_dir)
        self.n_node_thresholds = n_node_thresholds
        self.n_edge_thresholds = n_edge_thresholds
        
        # NOTE: thresholds needs to be sorted.
        self.n_node_thresholds.sort()
        self.n_edge_thresholds.sort()
        
        self.mol_id_col = mol_id_col
        self.mol_col = mol_col
        self.bond_multiplier = bond_multiplier # Because of directed edges.

    def serialize_hparams(self):
        """
        returns dictionary with all hyperparameters that will be saved. self.working_dir will be added
        to the dict in self.save_hparams.
        """
        return {'n_node_thresholds' : self.n_node_thresholds,
                'n_edge_thresholds' : self.n_edge_thresholds}
    
    @staticmethod
    def _test_small(x, n_node_threshold, n_edge_threshold, bond_multiplier):
        num_atoms, num_bonds = get_num_atoms_and_bonds(x)        
        return num_atoms < n_node_threshold and bond_multiplier * num_bonds < n_edge_threshold

    def _postprocess(self, data):
        """
        """
        datas = {}
        _data = data.groupby(self.mol_id_col).first()[self.mol_col].copy()
        available_idx = _data.index
        for i in range(len(self.n_node_thresholds)):
            n_node_tr = self.n_node_thresholds[i]
            n_edge_tr = self.n_edge_thresholds[i]
            _name = 'node' + str(n_node_tr) + '_' + 'edge' + str(n_edge_tr)

            _idx = _data[_data.apply(lambda x: self._test_small(x, n_node_tr, n_edge_tr, self.bond_multiplier))].index
            _idx = _idx.intersection(available_idx)

            datas[_name] = data[(data[self.mol_id_col].isin(_idx))]
            print('Num of unique molecules in {}: {}'.format(_name, len(_idx)))
            print('Num of pairs in {}: {}'.format(_name, len(datas[_name])))
            available_idx = available_idx.difference(_idx)
            del _idx
        
        _name = 'reminder'
        datas[_name] = data[(data[self.mol_id_col].isin(available_idx))]
        print('Num of unique molecules in {}: {}'.format(_name, len(available_idx)))
        print('Num of pairs in {}: {}'.format(_name, len(datas[_name])))

        return datas

    def postprocess(self):
        data_train, data_valid, data_test = self.load_data()

        if not data_train.empty:
            datas_train = self._postprocess(data_train)
        else:
            datas_train = {'reminder' : data_train.copy()}
        for name in datas_train.keys():
            datas_train[name].to_csv(os.path.join(self.working_dir, 'data_train_' + name + '.csv'), sep=';', index = False, header = True)    

        if not data_valid.empty:
            datas_valid = self._postprocess(data_valid)
        else:
            datas_valid = {'reminder' : data_valid.copy()}
        for name in datas_valid.keys():
            datas_valid[name].to_csv(os.path.join(self.working_dir, 'data_valid_' + name + '.csv'), sep=';', index = False, header = True)

        if not data_test.empty:
            datas_test = self._postprocess(data_test)
        else:
            datas_test = {'reminder' : data_test.copy()}
        for name in datas_test.keys():
            datas_test[name].to_csv(os.path.join(self.working_dir, 'data_test_' + name + '.csv'), sep=';', index = False, header = True)

        self.save_hparams()
        return None
