import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch.nn.functional as F
from utils import serialize, unserialize, LadderSampler, generate_square_mask, reset_random_seed, generate_decoder_mask, get_visited_locs
from quadkey_utils import collect_fn_quadkey, collect_fn_neg_quadkey_included
from model import QuadKeyLocPredictor
from dataset import NegInclLSBNDataset
from quadkey_dataset import QuadKeyLBSNDataset
from tqdm import tqdm
import neg_sampler
import joblib
from near_loc_query import LocQuerySystem
import loss
import time as Time
from collections import Counter, namedtuple

MarkovChainModel = namedtuple('MarkovChainModel', ['mc', 'states', "state2idx", 'count', 'visited_locs'])


def evaluate(model, test_dataset, negative_sampler, region_processer, loc2quadkey, device, batch_size=32, num_neg=100, neg_given=False):
    model.eval()
    reset_random_seed(42)
    if neg_given:
        loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=lambda e: collect_fn_neg_quadkey_included(e, test_dataset, region_processer, loc2quadkey))
    else:
        raise NotImplementedError("must provide eval sorting samples.")
    cnt = Counter()
    array = np.zeros(num_neg + 1)
    with torch.no_grad():
        for _, (user, loc, time, region, trg, trg_reg, _, _, ds) in enumerate(loader):
            user = user.to(device)
            loc = loc.to(device)
            time = time.to(device)
            region = region.to(device)
            trg = trg.to(device)
            trg_reg = trg_reg.to(device)
            src_mask = pad_sequence([torch.zeros(e, dtype=torch.bool).to(device) for e in ds], batch_first=True,
                                    padding_value=True)
            att_mask = generate_square_mask(max(ds), device)
            output = model(user, loc, region, time, att_mask, src_mask, trg, trg_reg, None, ds=ds)
            idx = output.sort(descending=True, dim=0)[1]
            order = idx.topk(1, dim=0, largest=False)[1]
            cnt.update(order.squeeze().tolist())
    for k, v in cnt.items():
        array[k] = v
    # hit rate and NDCG
    hr = array.cumsum()
    ndcg = 1 / np.log2(np.arange(0, num_neg + 1) + 2)
    ndcg = ndcg * array
    ndcg = ndcg.cumsum() / hr.max()
    hr = hr / hr.max()
    return hr[:10], ndcg[:10]


def train(model, train_dataset, test_dataset, optimizer, loss_fn, negative_sampler, test_sampler, region_processer, loc2quadkey, device, num_neg=5, batch_size=64, 
          num_epochs=10, writer=None, save_path=None, batch_size_test=32, num_neg_test=100, test_neg_given=False, num_workers=5, save_results=''):
    f = open("results/loss_gowalla_unweighted.txt", 'wt')
    for epoch_idx in range(num_epochs):
        start_time = Time.time()
        running_loss = 0.
        processed_batch = 0.
        data_loader = DataLoader(train_dataset, sampler=LadderSampler(train_dataset, batch_size), num_workers=num_workers, batch_size=batch_size, 
                                 collate_fn=lambda e: collect_fn_quadkey(e, train_dataset, negative_sampler, region_processer, loc2quadkey, k=num_neg))
        num_batch = len(data_loader)
        print("=====epoch {:>2d}=====".format(epoch_idx + 1))
        batch_iterator = tqdm(enumerate(data_loader), total=len(data_loader), leave=True)

        model.train()
        for batch_idx, (user, loc, time, region, trg, trg_reg, trg_nov, sample_probs, ds) in batch_iterator:
            user = user.to(device)
            loc = loc.to(device)
            time = time.to(device)
            region = region.to(device)
            trg = trg.to(device)
            trg_reg = trg_reg.to(device)
            sample_probs = sample_probs.to(device)
            optimizer.zero_grad()
            src_mask = pad_sequence([torch.zeros(e, dtype=torch.bool).to(device) for e in ds], batch_first=True, padding_value=True)
            att_mask = generate_square_mask(max(ds), device)
            output = model(user, loc, region, time, att_mask, src_mask, trg, trg_reg, att_mask.repeat(num_neg + 1, 1))
            output = output.view(-1, loc.size(0), loc.size(1)).permute(2, 1, 0)
            pos_score, neg_score = output.split([1, num_neg], -1)
            loss = loss_fn(pos_score, neg_score, sample_probs)
            # use only new location for training
            #keep = pad_sequence([torch.tensor(e).to(device) for e in trg_nov], batch_first=True)
            keep = pad_sequence([torch.ones(e, dtype=torch.float32).to(device) for e in ds], batch_first=True)
            loss = torch.sum(loss * keep) / torch.sum(torch.tensor(ds).to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            processed_batch += 1
            batch_iterator.set_postfix_str(f"loss={loss.item():.4f}")

        epoch_time = Time.time() - start_time
        print("epoch {:>2d} completed.".format(epoch_idx + 1))
        print("time taken: {:.2f} sec".format(epoch_time))
        print("avg. loss: {:.4f}".format(running_loss / processed_batch))
        print("epoch={:d}, loss={:.4f}".format(epoch_idx + 1, running_loss / processed_batch), file=f)
    print("training completed!")
    f.close()
    print("")
    print("=====evaluation=====")
    hr, ndcg = evaluate(model, test_dataset, test_sampler, region_processer, loc2quadkey, device, batch_size_test, num_neg_test, neg_given=test_neg_given)
    print("Hit@5: {:.4f}, NDCG@5: {:.4f}".format(hr[4], ndcg[4]))
    print("Hit@10: {:.4f}, NDCG@10: {:.4f}".format(hr[9], ndcg[9]))
    if save_results:
        with open(save_results, 'wt') as f:
            print("Hit@5: {:.4f}, NDCG@5: {:.4f}".format(hr[4], ndcg[4]), file=f)
            print("Hit@10: {:.4f}, NDCG@10: {:.4f}".format(hr[9], ndcg[9]), file=f) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--eval_samples', type=str, default='')
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--load_path', type=str, default='')
    parser.add_argument('--results_path', type=str, default='')
    args = parser.parse_args()

    config = unserialize(args.config)

    filename_raw = os.path.join(args.dataset, "totalCheckins.txt")
    filename_clean = os.path.join(args.dataset, "QuadKeyLSBNDataset.data")
    user2mc_filename = os.path.join(args.dataset, "reg_trans_pmc_model.pkl")
    loc_query_tree_path = os.path.join(args.dataset, "loc_query_tree.pkl")
    knn_wrmf_sample_prob_path = os.path.join(args.dataset, "knn_wrmf_sample_prob.pkl")

    reset_random_seed(42)

    if not os.path.isfile(filename_clean):
        dataset = QuadKeyLBSNDataset(filename_raw)
        serialize(dataset, filename_clean)
    else:
        dataset = unserialize(filename_clean)

    if config["train"]["negative_sampler"] in {"PopularitySampler", "KNNPopularitySampler", "KNNSampler", "KNNWRMFSampler"}:
        user_visited_locs = get_visited_locs(dataset)

    train_dataset, test_dataset = dataset.split()
    region_processer = dataset.QUADKEY

    if args.eval_samples:
        eval_samples = np.load(args.eval_samples)
        test_dataset = NegInclLSBNDataset(test_dataset, eval_samples)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = QuadKeyLocPredictor(
        nuser=train_dataset.n_user,
        nloc=train_dataset.n_loc,
        ntime=train_dataset.n_time,
        nquadkey=len(region_processer.vocab.itos),
        user_dim=int(config['model']['user_embedding_dim']),
        loc_dim=int(config['model']['location_embedding_dim']),
        time_dim=int(config['model']['time_embedding_dim']),
        reg_dim=int(config['model']['region_embedding_dim']),
        nhid=int(config['model']['hidden_dim_encoder']),
        nhead_enc=int(config['model']['num_heads_encoder']),
        nhead_dec=int(config['model']['num_heads_decoder']),
        nlayers=int(config['model']['num_layers_encoder']),
        dropout=float(config['model']['dropout']),
        **config['model']['extra_config']
    )
    model.to(device)
    loss_fn = loss.__getattribute__(config['train']['loss'])(**config['train']['loss_config'])

    if config["train"]["negative_sampler"] == "UniformNegativeSampler":
        sampler = neg_sampler.UniformNegativeSampler(train_dataset.n_loc)
    elif config["train"]["negative_sampler"] == "RegionUniformSampler":
        user2mc = unserialize(user2mc_filename)
        sampler = neg_sampler.RegionUniformSampler(
            user2mc=user2mc,
            n_region=train_dataset.n_region,
            region2loc=train_dataset.region2loc,
            exclude_visited=True
        )
    elif config["train"]["negative_sampler"] == "RadiusSampler":
        loc_query_sys = LocQuerySystem()
        loc_query_sys.load(loc_query_tree_path)
        sampler = neg_sampler.RadiusSampler(
            n_loc=train_dataset.n_loc,
            query_sys=loc_query_sys,
            **config["train"]["negative_sampler_config"]
        )
    elif config["train"]["negative_sampler"] == "KNNSampler":
        loc_query_sys = LocQuerySystem()
        loc_query_sys.load(loc_query_tree_path)
        sampler = neg_sampler.KNNSampler(
            query_sys=loc_query_sys,
            user_visited_locs=user_visited_locs,
            **config["train"]["negative_sampler_config"]
        )
    
    elif config["train"]["negative_sampler"] == "KNNPopularitySampler":
        loc_query_sys = LocQuerySystem()
        loc_query_sys.load(loc_query_tree_path)
        sampler = neg_sampler.KNNPopularitySampler(
            query_sys=loc_query_sys,
            loc2freq=train_dataset.locidx2freq,
            user_visited_locs=user_visited_locs,
            **config["train"]["negative_sampler_config"]
        )       
    elif config["train"]["negative_sampler"] == "KNNWRMFSampler":
        loc_query_sys = LocQuerySystem()
        loc_query_sys.load(loc_query_tree_path)
        knn_wrmf_prob = joblib.load(knn_wrmf_sample_prob_path)
        sampler = neg_sampler.KNNWRMFSampler(
            query_sys=loc_query_sys,
            user_knn_probs=knn_wrmf_prob['user_knn_probs'],
            user_knn_cum_probs=knn_wrmf_prob['user_knn_cum_probs'],
            user_visited_locs=user_visited_locs,
            **config["train"]["negative_sampler_config"]
        )
    elif config["train"]["negative_sampler"] == "PopularitySampler":
        sampler = neg_sampler.PopularitySampler(
            loc2freq=train_dataset.locidx2freq,
            user_visited_locs=user_visited_locs,
            exclude_visited=True
        )

    if config['optimizer']['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config['optimizer']['learning_rate']), betas=(0.9, 0.98))

    test_sampler = neg_sampler.UniformNegativeSampler(train_dataset.n_loc) 

    if args.load_path:
        model.load(args.load_path)
        print("=====evaluation=====")
        hr, ndcg = evaluate(model, test_dataset, test_sampler, region_processer, dataset.loc2quadkey, device, config['test']['batch_size'], config['test']['num_negative_samples'], neg_given=bool(args.eval_samples))
        print("Hit@5: {:.4f}, NDCG@5: {:.4f}".format(hr[4], ndcg[4]))
        print("Hit@10: {:.4f}, NDCG@10: {:.4f}".format(hr[9], ndcg[9]))
        if args.results_path:
            with open(args.results_path, 'wt') as f:
                print("Hit@5: {:.4f}, NDCG@5: {:.4f}".format(hr[4], ndcg[4]), file=f)
                print("Hit@10: {:.4f}, NDCG@10: {:.4f}".format(hr[9], ndcg[9]), file=f)
        exit()
        
    train(
        model=model, 
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        optimizer=optimizer,
        loss_fn=loss_fn,
        negative_sampler=sampler,
        test_sampler=test_sampler,
        region_processer=region_processer,
        loc2quadkey=dataset.loc2quadkey,
        device=device,
        num_neg=config['train']['num_negative_samples'], 
        batch_size=config['train']['batch_size'],
        num_epochs=config['train']['num_epochs'],
        writer=None,
        batch_size_test=config['test']['batch_size'],
        num_neg_test=config['test']['num_negative_samples'],
        test_neg_given=bool(args.eval_samples),
        num_workers=config['train']['num_workers'],
        save_results=args.results_path
    )

    if args.save_path:
        model.save(args.save_path)