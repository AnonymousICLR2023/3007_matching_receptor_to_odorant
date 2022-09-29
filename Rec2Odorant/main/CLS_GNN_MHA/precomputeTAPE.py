import os
import json
import tables
import torch
import pandas
import re
import functools

from mol2graph.read import read_fasta

from Rec2Odorant.main.base_loader import BaseDataset, BaseDataLoader

class PrecomputeBertDataset(BaseDataset):
    """
    """
    def __init__(self, data, seq_col, id_col,
                 orient='columns'):
        self.seq_col = seq_col # 'seq'
        self.id_col = id_col
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        seq = self.data.iloc[index][self.seq_col]
        seq = ' '.join(list(seq))
        seq = re.sub(r"[UZOB]", "X", seq)

        ids = self.data.iloc[index][self.id_col]
        return ids, seq


def collate_fn(batch, tokenizer):
        """
        """
        ids, batch = zip(*batch) # transposed
        seqs = torch.tensor([tokenizer.encode(ele.replace(' ', '')) for ele in batch])
        return ids, seqs


class PrecomputeBertLoader(BaseDataLoader):
    """
    """
    def __init__(self, dataset, tokenizer,
                    batch_size=1,
                    n_partitions = 0,
                    shuffle=False, 
                    rng=None, 
                    drop_last=False):
        if batch_size > 1:
            raise NotImplementedError('Batch size needs to be 1 because of the problem with batch tokenizer for TAPE.')
        super(self.__class__, self).__init__(dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        rng = rng,
        drop_last = drop_last,
        collate_fn = functools.partial(collate_fn, tokenizer = tokenizer),
        )


class PrecomputeTapeBERT_CLS:
    def __init__(self, data_file, save_dir, save_folder_name = None, mode = 'a', id_col = 'UniProt ID', seq_col = 'seq', dbname = '', batch_size = 8, bert_model = None, tokenizer = None):
        """
        """
        if save_folder_name is None:
            save_folder_name = __class__.__name__

        self.data_file = data_file
        self.save_dir = save_dir
        self.save_dir = os.path.join(save_dir, save_folder_name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.id_col = id_col 
        self.seq_col = seq_col
        self.batch_size = batch_size
        self.bert_model = bert_model
        self.tokenizer = tokenizer

        self.db_id_len = 64
        self.dbname = dbname
        self.mode = mode

    def serialize_hparams(self):
        """
        returns dictionary with all hyperparameters that will be saved. self.save_dir will be added
        to the dict in self.save_hparams.
        """
        return {'batch_size' : str(self.batch_size),
                'bert_model' : self.bert_model.__class__.__name__,
                'tokenizer' : self.tokenizer.__class__.__name__}    

    def save_hparams(self):
        hparams = self.serialize_hparams()
        hparams.update({'data_file' : self.data_file,
                        'save_dir' : self.save_dir})
        with open(os.path.join(self.save_dir, 'hparams.json'), 'w') as outfile:
            json.dump(hparams, outfile)

    def create_h5file(self, expectedrows):
        # Database handling:
        class PrecomputeBERTtable(tables.IsDescription):
            id    = tables.StringCol(self.db_id_len)
            hidden_states = tables.Float32Col(shape = (5*768,))
            # test = tables.Float64Col()

        h5file = tables.open_file(os.path.join(self.save_dir, self.dbname), mode = self.mode, title="TapeBERT")
        group = h5file.create_group("/", name = 'bert', title = 'TapeBERTgroup')
        self.filters = tables.Filters(complevel = 1, complib = 'blosc')
        self.table = h5file.create_table(group, name = 'BERTtable', description = PrecomputeBERTtable, title = "TapeBERTtable",
                                        filters = self.filters, expectedrows = expectedrows)
        self.h5file = h5file
        print(h5file)
        return None

    def apply_model(self, inputs):
        # batch_token_idxs = [self._encode(tokens) for tokens in batch_tokens]
        # inputs = torch.tensor(batch_token_idxs) # .unsqueeze(0)
        embedding = self.bert_model(inputs)[2]
        # Get CLS embedding for more than one layer.
        return torch.cat([emb[:, 0, :].squeeze(1) for emb in embedding[8:]], -1).detach().numpy()

    def _precompute_and_save(self, data):
        dataset = PrecomputeBertDataset(data, seq_col = self.seq_col, id_col = self.id_col)
        loader = PrecomputeBertLoader(dataset, tokenizer = self.tokenizer, batch_size = self.batch_size,
                    n_partitions = 0, shuffle=False, rng=None, drop_last=False)

        row = self.table.row

        for i, batch in enumerate(loader):
            ids, batch = batch
            hidden_states = self.apply_model(batch)

            for j in range(len(ids)):
                if len(ids[j]) > self.db_id_len:
                    raise ValueError('ID "{}" is too long for db_id_len: {}'.format(ids[j], self.db_id_len))
                row['id'] = ids[j]
                row['hidden_states'] = hidden_states[j]
                # row['test'] = numpy.random.normal(size = (numpy.random.randint(low = 1, high = 100), 10))
                row.append()

            if i >= 10:
                self.table.flush()
        
        print('creating index...')
        self.table.cols.id.create_index(optlevel=9, kind='full', filters = self.filters) # Create index for finished table to speed up search
        self.table.flush()
        return None

    def load_data(self):
        _, ext = os.path.splitext(self.data_file)
        if ext == ".fasta" or ext == '.fa':
            df = read_fasta(self.data_file)
            df.name = self.seq_col
            df = df.to_frame()
            df.index.name = self.id_col
            df.reset_index(inplace = True)
        elif ext == '.csv':
            df = pandas.read_csv(self.data_file, sep = self.sep, index_col = None, header = 0, usecols = [self.id_col, self.seq_col])
        return df

    def precompute_and_save(self):
        data = self.load_data()

        data = data[[self.id_col, self.seq_col]]
        data = data[~data[self.id_col].duplicated()]

        print('Number of records to process:  {}'.format(len(data)))

        self.create_h5file(expectedrows = len(data))
        self._precompute_and_save(data)
        self.h5file.close()

        self.save_hparams()
        return None





if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    from tape import TAPETokenizer, ProteinBertModel
    
    tokenizer = TAPETokenizer()
    bert_model = ProteinBertModel.from_pretrained('bert-base', 
                                            output_attentions=False, 
                                            output_hidden_states = True)
    
    precomuteTapeBERT_CLS = PrecomputeTapeBERT(data_file = os.path.join('..', 'RawData', 'Sequence_unaligned.fa'),
                                        save_dir = os.path.join('BERT_GNN', 'Data', 'precompute_test'),
                                        mode = 'w',
                                        dbname = 'Sequence_unaligned_TapeBERT_CLS.h5',
                                        id_col = 'ORid',
                                        batch_size = 1,
                                        bert_model = bert_model,
                                        tokenizer = tokenizer,
                                        )
    
    precomuteTapeBERT_CLS.precompute_and_save()
    precomuteTapeBERT_CLS.h5file.close()