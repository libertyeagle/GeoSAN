from datetime import datetime
import numpy as np
from utils import latlon2quadkey
from torch.utils.data import Dataset
from collections import defaultdict
import copy
import math


class LBSNDataset(Dataset):
    def __init__(self, filename):
        self.loc2idx = {'<pad>': 0}
        self.loc2gps = {'<pad>': (0.0, 0.0)}
        self.idx2loc = {0: '<pad>'}
        # (latitude, longitude) tuple
        self.idx2gps = {0: (0.0, 0.0)}
        self.loc2count = {}
        self.n_loc = 1
        self.build_vocab(filename)
        print(f'{self.n_loc} locations')
        self.user_seq, self.user2idx, self.region2idx, self.n_user, self.n_region, self.region2loc, self.n_time = self.processing(filename)
        print(f'{len(self.user_seq)} users')
        print(f'{len(self.region2idx)} regions')

    def region_stats(self):
        num_reg_locs = []
        for reg in self.region2loc:
            num_reg_locs.append(len(self.region2loc[reg]))
        num_reg_locs = np.array(num_reg_locs, dtype=np.int32)
        print("min #loc/region: {:d}, with {:d} regions".format(np.min(num_reg_locs), np.count_nonzero(num_reg_locs == 1)))
        print("max #loc/region:", np.max(num_reg_locs))
        print("avg #loc/region: {:.4f}".format(np.mean(num_reg_locs)))
        hist, bin_edges = np.histogram(num_reg_locs, bins=[1, 3, 5, 10, 20, 50, 100, 200, np.max(num_reg_locs)])
        for i in range(len(bin_edges) - 1):
            print("#loc in [{}, {}]: {:d} regions".format(math.ceil(bin_edges[i]), math.ceil(bin_edges[i + 1] - 1), hist[i]))

    def build_vocab(self, filename, min_freq=10):
        for line in open(filename):
            line = line.strip().split('\t')
            loc = line[4]
            coordinate = line[2], line[3]
            self.add_location(loc, coordinate)
        if min_freq > 0:
            self.n_loc = 1
            self.loc2idx = {'<pad>': 0}
            self.idx2loc = {0: '<pad>'}
            self.idx2gps = {0: (0.0, 0.0)}
            for loc in self.loc2count:
                if self.loc2count[loc] >= min_freq:
                    self.add_location(loc, self.loc2gps[loc])
        self.locidx2freq = np.zeros(self.n_loc - 1, dtype=np.int32)
        for idx, loc in self.idx2loc.items():
            if idx != 0:
                self.locidx2freq[idx - 1] = self.loc2count[loc]

    def add_location(self, loc, coordinate):
        if loc not in self.loc2idx:
            self.loc2idx[loc] = self.n_loc
            self.loc2gps[loc] = coordinate
            self.idx2loc[self.n_loc] = loc
            self.idx2gps[self.n_loc] = coordinate
            if loc not in self.loc2count:
                self.loc2count[loc] = 1
            self.n_loc += 1
        else:
            self.loc2count[loc] += 1

    def processing(self, filename, min_freq=20):
        user_seq = {}
        user_seq_array = list()
        region2idx = {}
        idx2region = {}
        regidx2loc = defaultdict(set)
        n_region = 1
        user2idx = {}
        n_users = 1
        for line in open(filename):
            user, time, lat, lon, loc = line.strip().split('\t')
            if loc not in self.loc2idx:
                continue
            time = datetime.strptime(time, '%Y-%m-%dT%H:%M:%SZ')
            time_idx = time.weekday() * 24 + time.hour + 1
            loc_idx = self.loc2idx[loc]
            region = latlon2quadkey(float(lat), float(lon), 11)
            if region not in region2idx:
                region2idx[region] = n_region
                idx2region[n_region] = region
                n_region += 1
            region_idx = region2idx[region]
            regidx2loc[region_idx].add(loc_idx)
            if user not in user_seq:
                user_seq[user] = list()
            user_seq[user].append([loc_idx, time_idx, region_idx, time])
        for user, seq in user_seq.items():
            if len(seq) >= min_freq:
                user2idx[user] = n_users
                user_idx = n_users
                seq_new = list()
                tmp_set = set()
                cnt = 0
                for loc, t, region, _ in sorted(seq, key=lambda e: e[3]):
                    if loc in tmp_set:
                        seq_new.append((user_idx, loc, t, region, True))
                    else:
                        seq_new.append((user_idx, loc, t, region, False))
                        tmp_set.add(loc)
                        cnt += 1
                if cnt > min_freq / 2:
                    n_users += 1
                    user_seq_array.append(seq_new)
        return user_seq_array, user2idx, region2idx, n_users, n_region, regidx2loc, 169

    def __len__(self):
        return len(self.user_seq)

    def __getitem__(self, idx):
        return self.user_seq[idx]

    def split(self, max_len=100):
        train_ = copy.copy(self)
        test_ = copy.copy(self)
        train_seq = list()
        test_seq = list()
        for u in range(len(self)):
            seq = self[u]
            i = 0
            for i in reversed(range(len(seq))):
                if not seq[i][4]:
                    break
            for b in range(math.floor((i + max_len - 1) // max_len)):
                if (i - b * max_len) > max_len*1.1:
                    trg = seq[(i - (b + 1) * max_len): (i - b * max_len)]
                    src = seq[(i - (b + 1) * max_len - 1): (i - b * max_len - 1)]
                    train_seq.append((src, trg))
                else:
                    trg = seq[1: (i - b * max_len)]
                    src = seq[0: (i - b * max_len - 1)]
                    train_seq.append((src, trg))
                    break
            test_seq.append((seq[max(0, -max_len+i):i], seq[i:i+1]))
        train_.user_seq = train_seq
        test_.user_seq = sorted(test_seq, key=lambda e: len(e[0]))
        return train_, test_


class NegInclLSBNDataset(Dataset):
    def __init__(self, test_dataset, eval_sort_samples):
        self.user_seq = test_dataset.user_seq
        self.sort_samples = eval_sort_samples

    def __len__(self):
        return len(self.user_seq)

    def __getitem__(self, idx):
        return self.user_seq[idx][0], self.user_seq[idx][1], self.sort_samples[idx]