import torch
import torch.nn as nn
import random
import numpy as np
from tqdm import tqdm
import bisect

# class RegionUniformNegativeSampler(nn.Module):
#     def __init__(self, region2loc, nloc):
#         nn.Module.__init__(self)
#         self.reg2loc = region2loc
#         self.n_loc = nloc

#     def forward(self, trg_seq, k):
#         neg_samples = []
#         for check_in in trg_seq:
#             region = check_in[3]
#             locs_region = self.reg2loc[region]
#             num_loc_in_reg = len(locs_region) - 1
#             if num_loc_in_reg > k:
#                 samples = []
#                 for _ in range(k):
#                     sample = random.sample(locs_region, 1)
#                     while sample[0] == check_in[1]:
#                         sample = random.sample(locs_region, 1)
#                     samples.extend(sample)
#             else:
#                 samples = list(locs_region - {check_in[1]})
#                 samples.extend(np.random.randint(1, self.n_loc, size=k - num_loc_in_reg))
#             neg_samples.append(samples)
#         neg_samples = torch.tensor(neg_samples, dtype=torch.long)
#         return neg_samples, torch.zeros_like(neg_samples, dtype=torch.float32)


class UniformNegativeSampler(nn.Module):
    def __init__(self, nloc):
        nn.Module.__init__(self)
        self.n_loc = nloc
    
    def forward(self, trg_seq, k, **kwargs):
        return torch.randint(1, self.n_loc, [len(trg_seq), k]), torch.ones(len(trg_seq), k, dtype=torch.float32)


class PopularitySampler(nn.Module):
    def __init__(self, loc2freq, user_visited_locs, exclude_visited=False):
        nn.Module.__init__(self)
        self.loc2freq = np.log(loc2freq + 1)
        self.pop_prob = loc2freq / np.sum(loc2freq)
        self.pop_cum_prob = self.pop_prob.cumsum()
        self.user_visited_locs = user_visited_locs
        self.exclude_visited = exclude_visited
    
    def forward(self, trg_seq, k, user, **kwargs):
        neg_samples = []
        sample_probs = []
        for _ in range(len(trg_seq)):
            samples = []
            probs = []
            for _ in range(k):
                sample = bisect.bisect(self.pop_cum_prob, random.random())
                if self.exclude_visited:
                    while (sample + 1) in self.user_visited_locs[user]:
                        sample = bisect.bisect(self.pop_cum_prob, random.random())
                p = self.loc2freq[sample]
                samples.append(sample + 1)
                probs.append(p)
            neg_samples.append(samples)
            sample_probs.append(probs)
        neg_samples = torch.tensor(neg_samples, dtype=torch.long)
        probs = torch.tensor(sample_probs, dtype=torch.float32)
        return neg_samples, probs


class RegionUniformSampler(nn.Module):
    def __init__(self, user2mc, n_region, region2loc, exclude_visited=False):
        nn.Module.__init__(self)
        self.user2mc = user2mc
        self.region2loc = region2loc
        self.region2count = np.zeros(n_region, dtype=np.int32)
        for reg, locs in self.region2loc.items():
            self.region2count[reg] = len(locs)
        self.exclude_visited = exclude_visited
    
    def forward(self, trg_seq, k, user, **kwargs):
        mc = self.user2mc[user]
        mc_trans_probs = mc.mc
        states = mc.states
        user_states_count = mc.count
        user_visited_locs = mc.visited_locs
        state2idx = mc.state2idx
        total_states_count = self.region2count[states]

        # no possible locations to sample, fall back to uniform sampling
        if self.exclude_visited and np.all(total_states_count - user_states_count == 0):
            return torch.randint(1, self.n_loc, [len(trg_seq), k]), torch.ones(len(trg_seq), k, dtype=torch.float32)

        neg_samples = []
        samples_probs = []
        for check_in in trg_seq:
            target_region = check_in[3]
            if target_region in state2idx:
                target_state = state2idx[target_region]
                unnormalized_probs = mc_trans_probs[target_state, :]
                if not self.exclude_visited:
                    trans_prob = total_states_count * unnormalized_probs
                else:
                    trans_prob = (total_states_count - user_states_count) * unnormalized_probs
                
            else:
                # target location's region is in a new region, not included in Markov Chain
                # uniform sample regions
                unnormalized_probs = np.ones(len(states)) / len(states)
                if not self.exclude_visited:
                    trans_prob = total_states_count * unnormalized_probs
                else:
                    trans_prob = (total_states_count - user_states_count) * unnormalized_probs
            
            # in case all locations in all possible states have been visited by this user
            if self.exclude_visited:
                if np.sum(trans_prob) == 0 or np.all(total_states_count[trans_prob > 0] - user_states_count[trans_prob > 0] == 0):
                    unnormalized_probs = np.ones(len(states)) / len(states)
                    trans_prob = (total_states_count - user_states_count) * unnormalized_probs 

            trans_prob /= np.sum(trans_prob)
            cum_prob = trans_prob.cumsum()

            samples = []
            probs = []
            for _ in range(k):
                # use bisect_left so that we won't stuck in a region whose locations are all visited
                # by this user
                sampled_state = bisect.bisect_left(cum_prob, 1 - random.random())
                sampled_region = states[sampled_state]
                sampled_loc = random.sample(self.region2loc[sampled_region], 1)[0]
                if self.exclude_visited:
                    while total_states_count[sampled_state] - user_states_count[sampled_state] == 0:
                        sampled_state = bisect.bisect_left(cum_prob, 1 - random.random())
                    sampled_region = states[sampled_state]
                    while sampled_loc in user_visited_locs:
                        sampled_loc = random.sample(self.region2loc[sampled_region], 1)[0]
                samples.append(sampled_loc)
                probs.append(unnormalized_probs[sampled_state])
            neg_samples.append(samples)
            samples_probs.append(probs) 
       
        neg_samples = torch.tensor(neg_samples, dtype=torch.long)
        probs = torch.tensor(samples_probs, dtype=torch.float32)
        return neg_samples, probs 


class KNNSampler(nn.Module):
    def __init__(self, query_sys, user_visited_locs, num_nearest=100, exclude_visited=False):
        nn.Module.__init__(self)
        self.query_sys = query_sys
        self.num_nearest = num_nearest
        self.user_visited_locs = user_visited_locs
        self.exclude_visited = exclude_visited
        
    def forward(self, trg_seq, k, user, **kwargs):
        neg_samples = []
        for check_in in trg_seq:
            trg_loc = check_in[1]
            nearby_locs = self.query_sys.get_knn(trg_loc, k=self.num_nearest)
            if not self.exclude_visited:
                samples = np.random.choice(nearby_locs, size=k, replace=True)
            else:
                samples = []
                for _ in range(k):
                    sample = np.random.choice(nearby_locs)
                    while sample in self.user_visited_locs[user]:
                        sample = np.random.choice(nearby_locs)
                    samples.append(sample)
            neg_samples.append(samples)
        neg_samples = torch.tensor(neg_samples, dtype=torch.long)
        probs = torch.ones_like(neg_samples, dtype=torch.float32)
        return neg_samples, probs


class RadiusSampler(nn.Module):
    def __init__(self, n_loc, query_sys, radius=10.0, threshold=5, epsilon=1.0):
        nn.Module.__init__(self)
        self.n_loc = n_loc
        self.query_sys = query_sys
        self.radius = radius
        self.threshold = threshold
        self.epsilon = epsilon
    
    def forward(self, trg_seq, k, **kwargs):
        neg_samples = []
        sample_probs = []
        for check_in in trg_seq:
            trg_loc = check_in[1]
            nearby_locs = self.query_sys.get_radius(trg_loc, r=self.radius)
            nearby_locs_set = set(nearby_locs)
            nearby_count = len(nearby_locs)
            if nearby_count < self.threshold:
                samples = np.random.randint(1, self.n_loc, size=k, dtype=np.int32)
                if self.epsilon < 1.0:
                    probs = np.ones_like(samples, dtype=np.float32)
            else:
                if self.epsilon < 1.0:
                    samples = []
                    probs = []
                    for _ in range(k):
                        if random.random() > self.epsilon:
                            sampled_loc = np.random.randint(1, self.n_loc)
                        else:
                            sampled_loc = np.random.choice(nearby_locs)
                        samples.append(sampled_loc)
                        if sampled_loc in nearby_locs_set:
                            probs.append(self.epsilon * self.n_loc + (1 - self.epsilon) * nearby_count)
                        else:
                            probs.append((1 - self.epsilon) * nearby_count)
                else:
                    samples = np.random.choice(nearby_locs, size=k, replace=True)
            neg_samples.append(samples)
            if self.epsilon < 1.0:
                sample_probs.append(probs)
        neg_samples = torch.tensor(neg_samples, dtype=torch.long)
        if self.epsilon < 1.0:
            probs = torch.tensor(sample_probs, dtype=torch.float32)
        else:
            probs = torch.ones_like(neg_samples, dtype=torch.float32)
        return neg_samples, probs


class KNNPopularitySampler(nn.Module):
    def __init__(self, query_sys, loc2freq, user_visited_locs, num_nearest=100, exclude_visited=False):
        nn.Module.__init__(self)
        self.num_nearest = num_nearest
        self.nearby_locs_finder = query_sys.knn_results[:, :self.num_nearest]
        self.loc2freq = np.log(loc2freq + 1)
        self.pop_freqs = np.zeros_like(self.nearby_locs_finder, dtype=np.int32)
        self.pop_probs = np.zeros_like(self.nearby_locs_finder, dtype=np.float64) 
        self.pop_cum_prob = np.zeros_like(self.nearby_locs_finder, dtype=np.float64)
        print("building prob index...")
        for i in tqdm(range(self.nearby_locs_finder.shape[0]), leave=True):
            nearby_locs = self.nearby_locs_finder[i]
            self.pop_freqs[i] = np.array([self.loc2freq[x - 1] for x in nearby_locs])
            self.pop_probs[i] = self.pop_freqs[i] / np.sum(self.pop_freqs[i])
            self.pop_cum_prob[i] = self.pop_probs[i].cumsum()
        self.user_visited_locs = user_visited_locs
        self.exclude_visited = exclude_visited
        
    def forward(self, trg_seq, k, user, **kwargs):
        neg_samples = []
        sample_probs = []
        for check_in in trg_seq:
            trg_loc = check_in[1]
            nearby_locs = self.nearby_locs_finder[trg_loc - 1]
            samples = []
            probs = []
            for _ in range(k):
                sample = bisect.bisect_left(self.pop_cum_prob[trg_loc - 1], random.random(), hi=self.num_nearest - 1)
                if self.exclude_visited:
                    while nearby_locs[sample] in self.user_visited_locs[user]:
                        sample = bisect.bisect_left(self.pop_cum_prob[trg_loc - 1], random.random(), hi=self.num_nearest - 1)
                p = self.pop_freqs[trg_loc - 1, sample]
                samples.append(nearby_locs[sample])
                probs.append(p)
            neg_samples.append(samples)
            sample_probs.append(probs)
        neg_samples = torch.tensor(neg_samples, dtype=torch.long)
        probs = torch.tensor(sample_probs, dtype=torch.float32)
        return neg_samples, probs


class KNNWRMFSampler(nn.Module):
    def __init__(self, query_sys, user_visited_locs, user_knn_probs, user_knn_cum_probs, num_nearest=100, exclude_visited=False):
        nn.Module.__init__(self)
        self.num_nearest = num_nearest
        self.nearby_locs_finder = query_sys.knn_results[:, :self.num_nearest]
        self.user_knn_probs = user_knn_probs
        self.user_knn_cum_probs = user_knn_cum_probs
        self.user_visited_locs = user_visited_locs
        self.exclude_visited = exclude_visited
        
    def forward(self, trg_seq, k, user, **kwargs):
        neg_samples = []
        sample_probs = []
        for check_in in trg_seq:
            trg_loc = check_in[1]
            nearby_locs = self.nearby_locs_finder[trg_loc - 1]
            prob = self.user_knn_probs[user - 1][trg_loc]
            cum_prob = self.user_knn_cum_probs[user - 1][trg_loc]
            samples = []
            probs = []
            for _ in range(k):
                sample = bisect.bisect_left(cum_prob, random.random(), hi=self.num_nearest - 1)
                if self.exclude_visited:
                    while nearby_locs[sample] in self.user_visited_locs[user]:
                        sample = bisect.bisect_left(cum_prob, random.random(), hi=self.num_nearest - 1)
                p = prob[sample]
                samples.append(nearby_locs[sample])
                probs.append(p)
            neg_samples.append(samples)
            sample_probs.append(probs)
        neg_samples = torch.tensor(neg_samples, dtype=torch.long)
        probs = torch.tensor(sample_probs, dtype=torch.float32)
        return neg_samples, probs