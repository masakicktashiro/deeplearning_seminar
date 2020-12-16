import os
import random
from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
import sacrebleu

from models import *
from attn_models import *

class IWSLTDataset:

    def __init__(self, src_path, tgt_path, src_wd2id=None, tgt_wd2id=None, max_length=None):
        max_length = 2048 if max_length is None else max_length
        self.src, self.tgt = [], []
        with open(src_path, "r") as f:
            src = [i.split() for i in f.readlines()]
        with open(tgt_path, "r") as f:
            tgt = [i.split() for i in f.readlines()]
        self.src_wd2id, self.tgt_wd2id = (src_wd2id, tgt_wd2id) if src_wd2id is not None \
                                                                                        else self.create_vocab(src, tgt)
        for s, t in zip(tqdm(src), tgt):
            src_ids = []
            tgt_ids  =[]
            # filter sentence whose length > max_length
            if len(s) > max_length or len(t) > max_length:
                continue
            for src_wd in s:
                src_ids.append(self.src_wd2id.get(src_wd, self.src_wd2id["<unk>"]))
            for tgt_wd in t:
                tgt_ids.append(self.tgt_wd2id.get(tgt_wd, self.tgt_wd2id["<unk>"]))
            self.src.append(src_ids)
            self.tgt.append(tgt_ids)
        self.src = [[self.src_wd2id["<bos>"]]+ src + [self.src_wd2id["<eos>"]] for src in self.src]
        self.tgt = [[self.tgt_wd2id["<bos>"]] + tgt + [self.tgt_wd2id["<eos>"]] for tgt in self.tgt]
        data = sorted([(i,j) for i,j in zip(self.src, self.tgt)], key=lambda x:len(x[0]))
        self.src, self.tgt = list(zip(*data))

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]

    def create_vocab(self, src, tgt):
        src_wd2id, tgt_wd2id = \
            {"<pad>":0, "<bos>":1, "<eos>":2, "<unk>":3},  {"<pad>":0, "<bos>":1, "<eos>":2, "<unk>":3}
        for s in src:
            for src_wd in s:
                if src_wd not in src_wd2id:
                    src_wd2id[src_wd] = len(src_wd2id)
        for t in tgt:
            for tgt_wd in t:
                if tgt_wd not in tgt_wd2id:
                    tgt_wd2id[tgt_wd] = len(tgt_wd2id)
        return src_wd2id, tgt_wd2id

class CollateFn:
    
    def __init__(self, pad_token_id, is_train=True):
        self.pad_token_id = pad_token_id
        self.is_train = is_train
        
    def __call__(self, batch):
        if self.is_train:
            srcs, tgts = list(zip(*batch))
        else:
            srcs = batch
        max_len_srcs = max([len(src) for src in srcs])
        srcs = [src + [self.pad_token_id] * (max_len_srcs - len(src)) for src in srcs]
        srcs = torch.LongTensor(srcs)
        max_len_tgts = max([len(tgt) for tgt in tgts])
        tgts = [tgt + [self.pad_token_id] * (max_len_tgts - len(tgt)) for tgt in tgts]
        tgts = torch.LongTensor(tgts)
        return srcs, tgts

def create_logger(log_dir, exp_name):
    logger = getLogger(exp_name)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{log_dir}/{exp_name}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_model(args, src_vocab_size, tgt_vocab_size):
    if args.model_name == "simple_gru":
        encoder = SimpleGRUEncoder(src_vocab_size, args.encoder_emb_size,
                                args.encoder_hid_size, args.encoder_num_layers,
                                args.encoder_bidirectional, args.dropout)
        decoder = SimpleGRUDecoder(tgt_vocab_size, args.decoder_emb_size,
                                args.decoder_hid_size, args.decoder_num_layers,
                                args.dropout)
    elif args.model_name == "simple_lstm":
        encoder = SimpleLSTMEncoder(src_vocab_size, args.encoder_emb_size,
                                args.encoder_hid_size, args.encoder_num_layers,
                                args.encoder_bidirectional, args.dropout)
        decoder = SimpleLSTMDecoder(tgt_vocab_size, args.decoder_emb_size,
                                args.decoder_hid_size, args.decoder_num_layers,
                                args.dropout)
    elif args.model_name == "attn_rnn":
        encoder = SimpleGRUEncoder(src_vocab_size, args.encoder_emb_size,
                                args.encoder_hid_size, args.encoder_num_layers,
                                args.encoder_bidirectional, args.dropout)
        decoder = AttnGRUDecoder(tgt_vocab_size, args.decoder_emb_size,
                                args.decoder_hid_size,
                                args.encoder_hid_size * (1 + args.encoder_bidirectional) * \
                                args.encoder_num_layers,
                                args.decoder_num_layers, args.dropout)
    else:
        raise ValueError
    model = Seq2SeqModel(encoder, decoder, args.decoding_style)
    return model

def get_optimizer(args, model):
    if args.optimizer == "adam":
        return optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "sgd":
        return optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == "adadelta":
        return optim.Adadelta(model.parameters(), lr=args.lr)
    else:
        raise ValueError

def get_scheduler(args, optimizer, milestones=None):
    if args.scheduler == "exp":
        return optim.lr_scheduler.ExponentialLR(optimizer, args.gamma)
    elif args.scheduler == "multi_step":
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones, args.gamma)
    else:
        raise ValueError

def init_uni_weights(model, width):
    for name, param in model.named_parameters():
        nn.init.uniform_(param.data, -width, width)


def count_parameters(model):
    count =  sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {count} trainable parameters')

def ids2sentence(ids, id2wd):
    """
    ids : tensor
        (bs, length)
    id2wd : Dict

    Return
    sentences : List
        list of str
    """
    tokens = [[id2wd[_id.item()] for _id in _ids] + ["<eos>"] for _ids in ids]
    tokens = [_tokens[:_tokens.index("<eos>")] for _tokens in tokens]
    sentences = [" ".join(_tokens).replace("@@ ", "") for _tokens in tokens]
    return sentences

def calc_bleu(sys, refs):
    bleu = sacrebleu.corpus_bleu(sys, refs)
    return bleu.score
