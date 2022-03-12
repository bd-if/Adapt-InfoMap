# -*- coding: utf-8 -*- 
"""

@Project: FaceClustering
@Author: intellif
Create time: 2022-03-07 15:47

"""

import os
from time import time
import numpy as np
from tqdm import tqdm
import infomap
from utlis.faiss_knn import faiss_knn
from configs import config
import warnings
warnings.filterwarnings('ignore')


def outlier_detect(delta_p, window_size):
    omega = window_size
    z = np.zeros_like(delta_p, dtype=np.float32)
    for j in tqdm(range(delta_p.shape[1]-omega, -1, -1)):
        mu_test = np.mean(delta_p[:, j:j+omega], axis=1)
        mu_ref = np.mean(delta_p[:, j:], axis=1)
        sigma_ref = np.std(delta_p[:, j:], axis=1)
        q = j + (omega+1)//2
        z[:, q] = np.abs(mu_test - mu_ref) / sigma_ref
    q_star = np.argmax(z, axis=1)
    return q_star


class FaceMap():

    def __init__(self):
        self.omega = config.window_size
        self.topK = config.topK
        self.knn_path = config.knn_path
        self.label_path = config.label_path
        self.feat_path = config.feat_path
        self.feat_dim = config.feat_dim
        self.result_path = config.result_path
        os.makedirs(self.knn_path, exist_ok=True)
        os.makedirs(self.result_path, exist_ok=True)
        self._load_knn()
        self.t = time()

    def _load_knn(self):
        t0 = time()
        if os.path.exists(self.knn_path):
            knn = np.load(self.knn_path)
            knn = knn['data']
            if isinstance(knn, list):
                knn = np.array(knn)
            self.nbrs = knn[:, 0, :self.topK].astype(np.int32)
            self.sims = knn[:, 1, :self.topK].astype(np.float32)
        else:
            self.nbrs, self.sims = faiss_knn(self.feat_path, self.knn_path, self.feat_dim, self.topK)
        print('time cost of load knn: {:.2f}s'.format(time() - t0))

    def transition_prob_by_threshold(self, th=0.62):
        single, links, weights = [], [], []
        for i in tqdm(range(self.nbrs.shape[0])):
            c = 0
            for j, nbr in enumerate(self.nbrs[i]):
                if self.sims[i, j] >= th:
                    c += 1
                    links.append((i, nbr))
                    weights.append(self.sims[i, j])
                else:
                    break
            if c == 0:
                single.append(i)
        self.links = np.array(links, dtype=np.uint32)
        self.weights = np.array(weights, dtype=np.float32)
        self.single = np.array(single, dtype=np.uint32)

    def adjust_transition_prob(self):
        p = self.sims / np.sum(self.sims, axis=1, keepdims=True)
        t0 = time()
        delta_p = p[:, :-1] - p[:, 1:]
        q = outlier_detect(delta_p, self.omega)
        print('time cost of outlier_detect: {:.2f}s'.format(time() - t0))
        
        single, links, weights = [], [], []
        for i, k in enumerate(q):
            count = 0
            for idx, j in enumerate(self.nbrs[i, :k+1]):
                if i == j:
                    pass
                else:
                    count += 1
                    links.append((i, j))
                    weights.append(p[i, idx])
            if count == 0:
                single.append(i)
        self.links = np.array(links, dtype=np.uint32)
        self.weights = np.array(weights, dtype=np.float32)
        self.single = np.array(single, dtype=np.uint32)
    
    def face_cluster(self):
        info = infomap.Infomap("--two-level", flow_model='undirected')
        for (i, j), sim in tqdm(zip(self.links, self.weights)):
            _ = info.addLink(i, j, sim)
        del self.links
        del self.weights

        info.run(seed=100)

        lb2idx = {}
        self.idx2lb = {}
        for node in info.iterTree():
            if node.moduleIndex() not in lb2idx:
                lb2idx[node.moduleIndex()] = []
            lb2idx[node.moduleIndex()].append(node.physicalId)

        for k, v in lb2idx.items():
            if k == 0:
                lb2idx[k] = v[2:]
                for u in v[2:]:
                    self.idx2lb[u] = k
            else:
                lb2idx[k] = v[1:]
                for u in v[1:]:
                    self.idx2lb[u] = k

        lb_len = len(lb2idx)
        if len(self.single) > 0:
            for k in self.single:
                if k in self.idx2lb:
                    continue
                self.idx2lb[k] = lb_len
                lb2idx[lb_len] = [k]
                lb_len += 1
        print('time cost of FaceMap: {:.2f}s'.format(time() - self.t))

        pred_labels = np.zeros(len(self.idx2lb)) - 1
        for k, v in self.idx2lb.items():
            pred_labels[k] = v
        np.save(self.result_path, pred_labels)
