from sklearn.neighbors import BallTree
import numpy as np
from tqdm import tqdm
import os 
import argparse
from dataset import LBSNDataset
import math
import json
try:
    import cPickle as _pickle
except ImportError:
    import pickle as _pickle


def serialize(obj, path, in_json=False):
    if in_json:
        with open(path, "w") as file:
            json.dump(obj, file, indent=2)
    else:
        with open(path, 'wb') as file:
            _pickle.dump(obj, file)

def unserialize(path):
    suffix = os.path.basename(path).split(".")[-1]
    if suffix == "json":
        with open(path, "r") as file:
            return json.load(file)
    else:
        with open(path, 'rb') as file:
            return _pickle.load(file)


class LocQuerySystem:
    def __init__(self):
        self.coordinates = []
        self.tree = None
        self.knn = None
        self.knn_results = None
        self.radius = None
        self.radius_results = None

    def build_tree(self, dataset):
        self.coordinates = np.zeros((len(dataset.idx2gps) - 1, 2), dtype=np.float64)
        for idx, (lat, lon) in dataset.idx2gps.items():
            if idx != 0:
                self.coordinates[idx - 1] = [lat, lon]
        self.tree = BallTree(
            self.coordinates,
            leaf_size=1,
            metric='haversine'
        )
    
    def prefetch_knn(self, k=100):
        self.knn = k
        self.knn_results = np.zeros((self.coordinates.shape[0], k), dtype=np.int32)
        for idx, gps in tqdm(enumerate(self.coordinates), total=len(self.coordinates), leave=True):
            trg_gps = gps.reshape(1, -1)
            _, knn_locs = self.tree.query(trg_gps, k + 1)
            knn_locs = knn_locs[0, 1:]
            knn_locs += 1
            self.knn_results[idx] = knn_locs
    
    def prefetch_radius(self, radius=10.0):
        self.radius = radius
        self.radius_results = {}
        radius /= 6371000/1000
        for idx, gps in tqdm(enumerate(self.coordinates), total=len(self.coordinates), leave=True):
            trg_gps = gps.reshape(1, -1)
            nearby_locs = self.tree.query_radius(trg_gps, r=radius)
            nearby_locs = nearby_locs[0]
            nearby_locs = np.delete(nearby_locs, np.where(nearby_locs == idx))
            nearby_locs += 1
            self.radius_results[idx + 1] = nearby_locs

    def get_knn(self, trg_loc, k=100):
        if k <= self.knn:
            return self.knn_results[trg_loc - 1][:k]
        trg_gps = self.coordinates[trg_loc - 1].reshape(1, -1)
        _, knn_locs = self.tree.query(trg_gps, k + 1)
        knn_locs = knn_locs[0, 1:]
        knn_locs += 1
        return knn_locs

    def get_radius(self, trg_loc, r=10.0):
        if r == self.radius:
            return self.radius_results[trg_loc]
        r /= 6371000/1000
        trg_gps = self.coordinates[trg_loc - 1].reshape(1, -1)
        nearby_locs = self.tree.query_radius(trg_gps, r=r)
        nearby_locs = nearby_locs[0]
        nearby_locs = np.delete(nearby_locs, np.where(nearby_locs == trg_loc - 1))
        nearby_locs += 1
        return nearby_locs

    def radius_stats(self, radius=10):
        radius /= 6371000/1000
        num_nearby_locs = []
        for gps in tqdm(self.coordinates, total=len(self.coordinates), leave=True):
            trg_gps = gps.reshape(1, -1)
            count = self.tree.query_radius(trg_gps, r=radius, count_only=True)[0]
            num_nearby_locs.append(count)
        num_nearby_locs = np.array(num_nearby_locs, dtype=np.int32)
        max_loc_idx = np.argsort(-num_nearby_locs)[0]
        print("max #nearby_locs: {:d}, at loc {:d}".format(num_nearby_locs[max_loc_idx], max_loc_idx + 1))
        #print("min #nearby_locs/: {:d}, with {:d} locs".format(np.min(num_nearby_locs), np.count_nonzero(num_nearby_locs == np.min(num_nearby_locs))))
        #print("min #nearby_locs/: {:d}, with {:d} locs".format(np.max(num_nearby_locs), np.count_nonzero(num_nearby_locs == np.max(num_nearby_locs))))
        #print("avg #nearby_locs: {:.4f}".format(np.mean(num_nearby_locs)))
        #hist, bin_edges = np.histogram(num_nearby_locs, bins=[1, 3, 5, 10, 20, 40, 100, 200, np.max(num_nearby_locs)])
        #for i in range(len(bin_edges) - 1):
        #    print("#nearby_locs in [{}, {}]: {:d}".format(math.ceil(bin_edges[i]), math.ceil(bin_edges[i + 1] - 1), hist[i]))
    
    def save(self, path):
        data = {
            'coordinates': self.coordinates,
            'tree': self.tree,
            'knn': self.knn,
            'knn_results': self.knn_results,
            'radius': self.radius,
            'radius_results': self.radius_results
        }
        serialize(data, path)
    
    def load(self, path):
        data = unserialize(path)
        self.coordinates = data['coordinates']
        self.tree = data['tree']
        self.knn = data['knn']
        self.knn_results = data['knn_results']
        self.radius = data['radius']
        self.radius_results = data['radius_results']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    filename_raw = os.path.join(args.dataset, "totalCheckins.txt")
    filename_clean = os.path.join(args.dataset, "LSBNDataset.data")
    save_path = os.path.join(args.dataset, "loc_query_tree.pkl")

    if not os.path.isfile(filename_clean):
        dataset = LBSNDataset(filename_raw)
        serialize(dataset, filename_clean)
    else:
        dataset = unserialize(filename_clean)    

    loc_query = LocQuerySystem()
    loc_query.build_tree(dataset)
    #loc_query.radius_stats(radius=10)
    loc_query.save(save_path)