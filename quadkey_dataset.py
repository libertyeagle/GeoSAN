from dataset import LBSNDataset
from datetime import datetime
import numpy as np
from utils import latlon2quadkey
from collections import defaultdict
from nltk import ngrams
from torchtext.data import Field

from utils import serialize, unserialize
import argparse
import os

LOD = 17

class QuadKeyLBSNDataset(LBSNDataset):
    def __init__(self, filename):
        super().__init__(filename)

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
            line = line.strip().split('\t')
            if len(line) < 5:
                continue
            #user, time, lat, lon, loc = line
            user, loc, lat, lon, time = line
            if loc not in self.loc2idx:
                continue
            #time = datetime.strptime(time, '%Y-%m-%dT%H:%M:%SZ')
            time = datetime.strptime(time, "%Y-%m-%d %H:%M:%S+00:00")
            time_idx = time.weekday() * 24 + time.hour + 1
            loc_idx = self.loc2idx[loc]
            region = latlon2quadkey(float(lat), float(lon), LOD)
            if region not in region2idx:
                region2idx[region] = n_region
                idx2region[n_region] = region
                n_region += 1
            region_idx = region2idx[region]
            regidx2loc[region_idx].add(loc_idx)
            if user not in user_seq:
                user_seq[user] = list()
            user_seq[user].append([loc_idx, time_idx, region_idx, region, time])
        for user, seq in user_seq.items():
            if len(seq) >= min_freq:
                user2idx[user] = n_users
                user_idx = n_users
                seq_new = list()
                tmp_set = set()
                cnt = 0
                for loc, t, _, region_quadkey, _ in sorted(seq, key=lambda e: e[4]):
                    if loc in tmp_set:
                        seq_new.append((user_idx, loc, t, region_quadkey, True))
                    else:
                        seq_new.append((user_idx, loc, t, region_quadkey, False))
                        tmp_set.add(loc)
                        cnt += 1
                if cnt > min_freq / 2:
                    n_users += 1
                    user_seq_array.append(seq_new)

        all_quadkeys = []
        for u in range(len(user_seq_array)):
            seq = user_seq_array[u]
            for i in range(len(seq)):
                check_in = seq[i]
                region_quadkey = check_in[3]
                region_quadkey_bigram = ' '.join([''.join(x) for x in ngrams(region_quadkey, 6)])
                region_quadkey_bigram = region_quadkey_bigram.split()
                all_quadkeys.append(region_quadkey_bigram)
                user_seq_array[u][i] = (check_in[0], check_in[1], check_in[2], region_quadkey_bigram, check_in[4])

        self.loc2quadkey = ['NULL']
        for l in range(1, self.n_loc):
            lat, lon = self.idx2gps[l]
            quadkey = latlon2quadkey(float(lat), float(lon), LOD)
            quadkey_bigram = ' '.join([''.join(x) for x in ngrams(quadkey, 6)])
            quadkey_bigram = quadkey_bigram.split()
            self.loc2quadkey.append(quadkey_bigram)
            all_quadkeys.append(quadkey_bigram)
        
        self.QUADKEY = Field(
            sequential=True,
            use_vocab=True,
            batch_first=True,
            unk_token=None,
            preprocessing=str.split
        )
        self.QUADKEY.build_vocab(all_quadkeys)

        return user_seq_array, user2idx, region2idx, n_users, n_region, regidx2loc, 169

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    filename_raw = os.path.join(args.dataset, "totalCheckins.txt")
    filename_clean = os.path.join(args.dataset, "QuadKeyLSBNDataset.data")

    if not os.path.isfile(filename_clean):
        dataset = QuadKeyLBSNDataset(filename_raw)
        serialize(dataset, filename_clean)
    else:
        dataset = unserialize(filename_clean)
    
    count = 0
    length = []
    for seq in dataset.user_seq:
        count += len(seq)
        length.append(len(seq))
    print("#check-ins:", count)
    print("#users:", dataset.n_user - 1)
    print("#locations:", dataset.n_loc - 1)
    print("#median seq len:", np.median(np.array(length)))