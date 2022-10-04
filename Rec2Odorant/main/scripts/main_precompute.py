import os
import argparse

from transformers import BertTokenizer, BertConfig, FlaxBertModel

from Rec2Odorant.main.CLS_GNN_MHA.precompute import PrecomputeProtBERT_CLS

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, required=True,
                        help='config file path')
    parser.add_argument('--cuda_device', type=int,
                        help='Set environment variable CUDA_VISIBLE_DEVICES')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Path to a directory where output is saved')

    args = parser.parse_args()
    print('Seqs file: {}'.format(args.data_file))
    print('Save dir: {}'.format(args.save_dir))
    print('---------------')

    # Without this, there is CUDA out of memory:
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

    # Set visible devices:
    if args.cuda_device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
        print('Setting CUDA_VISIBLE_DEVICES to: {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
    

    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    config = BertConfig.from_pretrained("Rostlab/prot_bert", output_hidden_states=True, output_attentions=False)
    bert_model = FlaxBertModel.from_pretrained("Rostlab/prot_bert", from_pt = True, config = config)

    precomuteBERT_CLS = PrecomputeProtBERT_CLS(data_file = args.data_file,
                                        save_dir = args.save_dir,
                                        mode = 'w',
                                        dbname = 'ProtBERT_CLS.h5',
                                        id_col = 'seq_id',
                                        seq_col = 'mutated_Sequence',
                                        batch_size = 4,
                                        bert_model = bert_model,
                                        tokenizer = tokenizer,
                                        )
    
    precomuteBERT_CLS.precompute_and_save()
    precomuteBERT_CLS.h5file.close()