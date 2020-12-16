import os
import argparse

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', default=False, type=bool)
    parser.add_argument('--verbose', default=0, type=int)
    
    parser.add_argument('--model_name', default="simple_gru", type=str)
    parser.add_argument('--batch_size', default=80, type=int)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--shuffle_data', default=False, type=bool)
    parser.add_argument('--log_dir',
            default=os.getcwd() + "/logs", type=str)
    parser.add_argument('--data_dir',
            default=os.getcwd() + "/data/iwslt14.tokenized.de-en", type=str)
    parser.add_argument('--save_dir',
            default=os.getcwd() + "/trained_models", type=str)
    parser.add_argument('--n_epoch', default=20, type=int)
    parser.add_argument('--teacher_forcing_ratio', default=1.0, type=float)
    parser.add_argument('--decoding_style', default="greedy", type=str)
    parser.add_argument('--patience', default=3, type=int)
    parser.add_argument('--optimizer', default="adam", type=str)
    parser.add_argument('--scheduler', default="exp", type=str)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--weight_decay', default=0.001, type=float)
    parser.add_argument('--gamma', default=0.5, type=float)
    
    parser.add_argument('--encoder_emb_size', default=256, type=int)
    parser.add_argument('--encoder_hid_size', default=256, type=int)
    parser.add_argument('--encoder_num_layers', default=1, type=int)
    parser.add_argument('--encoder_bidirectional', default=False, type=bool)
    parser.add_argument('--decoder_emb_size', default=512, type=int)
    parser.add_argument('--decoder_hid_size', default=512, type=int)
    parser.add_argument('--decoder_num_layers', default=1, type=int)
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--init_param_width', default=0, type=float)
    args = parser.parse_args()
    return args
