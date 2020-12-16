import warnings
import math
import codecs
import datetime
import functools
import os
from contextlib import redirect_stdout

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from configs import get_config
from utils import *
from models import Seq2SeqModel, SimpleGRUEncoder, SimpleGRUDecoder

def main():
    seed_everything()
    args = get_config()
    current_time = str(datetime.datetime.now()).replace(" ","_")
    exp_name = f"{current_time}_{args.model_name}"
    print(args, file=codecs.open(f"{args.log_dir}/{exp_name}.log", "w", "utf-8"))
    logger = create_logger(f"{args.log_dir}", exp_name)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    pad_token_id = 0
    data_dir = args.data_dir
    train_ds = IWSLTDataset(f"{data_dir}/train.de", f"{data_dir}/train.en", max_length=128)
    src_wd2id, tgt_wd2id = train_ds.src_wd2id, train_ds.tgt_wd2id
    tgt_id2wd = {j:i for i, j in tgt_wd2id.items()}
    val_ds = IWSLTDataset(f"{data_dir}/valid.de", f"{data_dir}/valid.en",
                            src_wd2id=src_wd2id, tgt_wd2id=tgt_wd2id)
    collate_fn = CollateFn(0, True)
    if args.debug:
        train_ds = train_ds[:400]
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, num_workers=2,
                        shuffle=args.shuffle_data, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=2,
                        collate_fn=collate_fn, shuffle=False)
    train_step = math.ceil(len(train_ds) / args.batch_size)
    src_vocab_size = len(src_wd2id)
    tgt_vocab_size = len(tgt_wd2id)
    model = get_model(args, src_vocab_size, tgt_vocab_size)
    if args.init_param_width != 0:
        init_fn = functools.partial(init_uni_weights, width=args.init_param_width)
        model.apply(init_fn)
    count_parameters(model)
    model = model.to(device)
    optimizer = get_optimizer(args, model) # optim.Adam(model.parameters(), lr=args.lr) 
    if args.scheduler == "mutli_step":
        milestones = [int(train_step * i) for i in range(5, args.n_epoch, 0.5)]
    else:
        milestones = None
    scheduler = get_scheduler(args, optimizer, milestones)
    train_fn(train_dl, val_dl, model, optimizer, scheduler,
            device, logger, tgt_id2wd, args=args) 

def train_fn(train_dl ,val_dl, model, optimizer, scheduler, device,
             logger, tgt_id2wd, args, model_name="best_model.bin", ignore_index=0):
    # criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    criterion = nn.NLLLoss(ignore_index=ignore_index)
    best_loss = 1e9
    cnt = 0
    vocab_size = model.decoder.vocab_size
    for epoch in range(args.n_epoch):
        model.train()
        losses = []
        for srcs, tgts in tqdm(train_dl):
            srcs = srcs.to(device)
            tgts = tgts.to(device)
            output = model(srcs, tgts[:, :-1], args.teacher_forcing_ratio)
            loss = criterion(output.reshape(-1, vocab_size), tgts[:, 1:].flatten())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            losses.append(loss.item())
            del loss, srcs, tgts, output
            torch.cuda.empty_cache()
        loss = np.mean(losses)
        lr = optimizer.param_groups[0]["lr"]
        logger.info(f"train {epoch} || loss : {loss} || PPL {2 ** loss} || lr {lr}")
        model.eval()
        losses = []
        scores = []
        for srcs, tgts in tqdm(val_dl):
            srcs = srcs.to(device)
            tgts = tgts.to(device)
            with torch.no_grad():
                output = model(srcs, tgts[:, :-1], args.teacher_forcing_ratio)
                pred = model.decode(srcs).detach().cpu()
            loss = criterion(output.reshape(-1, vocab_size), tgts[:, 1:].flatten())
            losses.append(loss.item())
            ref = ids2sentence(tgts[:, 1:].detach().cpu(), tgt_id2wd)
            sys = ids2sentence(pred, tgt_id2wd)
            with redirect_stdout(open(os.devnull, 'w')):
                # tokenizeに関するエラーを一時的に無視
                score = calc_bleu(sys, [ref])
            scores.append(score)
            if args.verbose:
                logger.info("sys :" + sys[0])
                logger.info("ref :" + ref[0])
        loss = np.mean(losses)
        score = np.mean(scores)
        logger.info(f"val {epoch} || loss : {loss} || PPL {2 ** loss}"+\
                    f"|| {score}")
        if loss < best_loss:
            best_loss = loss
            cnt = 0
            torch.save(model.state_dict(), f"{args.save_dir}/{model_name}")
        else:
            cnt += 1
            if cnt > args.patience:
                break

if __name__ == "__main__":
    main()
