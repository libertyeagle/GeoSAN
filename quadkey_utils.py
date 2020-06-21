import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np


def collect_fn_quadkey(batch, data_source, sampler, region_processer, loc2quadkey=None, k=5, with_trg_quadkey=True):
    src, trg = zip(*batch)
    user, loc, time, region = [], [], [], []
    data_size = []
    trg_ = []
    trg_probs_= []
    for e in src:
        u_, l_, t_, r_, b_ = zip(*e)
        data_size.append(len(u_))
        user.append(torch.tensor(u_))
        loc.append(torch.tensor(l_))
        time.append(torch.tensor(t_))
        # (L, LEN_QUADKEY)
        r_ = region_processer.numericalize(list(r_))
        region.append(r_)
    user_ = pad_sequence(user, batch_first=True)
    loc_ = pad_sequence(loc, batch_first=True)
    time_ = pad_sequence(time, batch_first=True)
    # (T, N, LEN_QUADKEY)
    region_ = pad_sequence(region, batch_first=False)
    if with_trg_quadkey:
        batch_trg_regs = []
        for i, seq in enumerate(trg):
            pos = torch.tensor([[e[1]] for e in seq])
            neg, probs = sampler(seq, k, user=seq[0][0])
            # (L, k+1)
            trg_seq = torch.cat([pos, neg], dim=-1)
            trg_.append(trg_seq)
            trg_regs = []
            for l in range(trg_seq.size(0)):
                regs = []
                for loc in trg_seq[l]:
                    regs.append(loc2quadkey[loc])
                trg_regs.append(region_processer.numericalize(regs))
            batch_trg_regs.append(torch.stack(trg_regs))
            trg_probs_.append(probs)
        # (N, T, k+1, LEN_QUADKEY)
        batch_trg_regs = pad_sequence(batch_trg_regs, batch_first=True)
        # [(1+k) * T, N, LEN_QUADKEY)
        batch_trg_regs = batch_trg_regs.permute(2, 1, 0, 3).contiguous().view(-1, batch_trg_regs.size(0), batch_trg_regs.size(3))
        trg_ = pad_sequence(trg_, batch_first=True)
        trg_probs_ = pad_sequence(trg_probs_, batch_first=True, padding_value=1.0)
        trg_ = trg_.permute(2, 1, 0).contiguous().view(-1, trg_.size(0))
        trg_nov_ = [[not e[-1] for e in seq] for seq in trg]    
        return user_.t(), loc_.t(), time_.t(), region_, trg_, batch_trg_regs, trg_nov_, trg_probs_, data_size
    else:
        for i, seq in enumerate(trg):
            pos = torch.tensor([[e[1]] for e in seq])
            neg, probs = sampler(seq, k, user=seq[0][0])
            trg_.append(torch.cat([pos, neg], dim=-1))
            trg_probs_.append(probs)
        trg_ = pad_sequence(trg_, batch_first=True)
        trg_probs_ = pad_sequence(trg_probs_, batch_first=True, padding_value=1.0)
        trg_ = trg_.permute(2, 1, 0).contiguous().view(-1, trg_.size(0))
        trg_nov_ = [[not e[-1] for e in seq] for seq in trg]    
        return user_.t(), loc_.t(), time_.t(), region_, trg_, trg_nov_, trg_probs_, data_size 


def collect_fn_neg_quadkey_included(batch, data_source, region_processer, loc2quadkey=None, with_trg_quadkey=True):
    src, trg, neg_samples = zip(*batch)
    user, loc, time, region = [], [], [], []
    data_size = []
    trg_ = []
    trg_probs_= []
    for e in src:
        u_, l_, t_, r_, b_ = zip(*e)
        data_size.append(len(u_))
        user.append(torch.tensor(u_))
        loc.append(torch.tensor(l_))
        time.append(torch.tensor(t_))
        r_ = region_processer.numericalize(list(r_))
        region.append(r_)
    user_ = pad_sequence(user, batch_first=True)
    loc_ = pad_sequence(loc, batch_first=True)
    time_ = pad_sequence(time, batch_first=True)
    region_ = pad_sequence(region, batch_first=False)
    if not with_trg_quadkey:
        for seq, neg in zip(trg, neg_samples):
            pos = torch.tensor([[e[1]] for e in seq])
            neg = torch.tensor(np.expand_dims(neg, axis=0), dtype=torch.long)
            trg_.append(torch.cat([pos, neg], dim=-1))
        trg_ = pad_sequence(trg_, batch_first=True)
        trg_ = trg_.permute(2, 1, 0).contiguous().view(-1, trg_.size(0))
        return user_.t(), loc_.t(), time_.t(), region_, trg_, None, None, data_size
    else:
        batch_trg_regs = []
        for seq, neg in zip(trg, neg_samples):
            pos = torch.tensor([[e[1]] for e in seq])
            neg = torch.tensor(np.expand_dims(neg, axis=0), dtype=torch.long)
            trg_seq = torch.cat([pos, neg], dim=-1)
            trg_.append(trg_seq)
            trg_regs = []
            for l in range(trg_seq.size(0)):
                regs = []
                for loc in trg_seq[l]:
                    regs.append(loc2quadkey[loc])
                trg_regs.append(region_processer.numericalize(regs))
            batch_trg_regs.append(torch.stack(trg_regs))
        batch_trg_regs = pad_sequence(batch_trg_regs, batch_first=True)
        batch_trg_regs = batch_trg_regs.permute(2, 1, 0, 3).contiguous().view(-1, batch_trg_regs.size(0), batch_trg_regs.size(3))
        trg_ = pad_sequence(trg_, batch_first=True)
        trg_ = trg_.permute(2, 1, 0).contiguous().view(-1, trg_.size(0))
        return user_.t(), loc_.t(), time_.t(), region_, trg_, batch_trg_regs, None, None, data_size