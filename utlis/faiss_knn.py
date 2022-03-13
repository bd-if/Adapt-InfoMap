# -*- coding: utf-8 -*-

import numpy as np
import faiss
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def l2norm(vec):
    vec /= np.linalg.norm(vec, axis=1).reshape(-1, 1)
    return vec


def load_feat(feat_path, feat_dim=256):
    if '.npy' in feat_path:
        feat = np.load(feat_path).astype(np.float32)
    else:
        feat = np.fromfile(feat_path, dtype=np.float32)
        feat = feat.reshape(-1, feat_dim)
    return feat


def faiss_knn(feat_path, knn_path, feat_dim, k=256):
    feat = load_feat(feat_path, feat_dim)
    print('features shape:', feat.shape)
    feat = l2norm(feat)

    index = faiss.IndexFlatIP(feat_dim)

    # use single gpu
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index)

    # use all gpus
    # index = faiss.index_cpu_to_all_gpus(index)

    index.add(feat)
    batch_size = 200000
    n = int(np.ceil(feat.shape[0] / batch_size))
    sims = np.array([], dtype=np.float16).reshape(-1, k+1)
    nbrs = np.array([], dtype=np.uint32).reshape(-1, k+1)

    for i in tqdm(range(n)):
        start = i * batch_size
        end = (i+1) * batch_size
        query = feat[start:end]
        sim, nbr = index.search(query, k+1)
        sims = np.vstack((sims, sim))
        nbrs = np.vstack((nbrs, nbr))

    # remove itself
    for i in range(nbrs.shape[0]):
        if i == nbrs[i, 0]:
            pass
        else:
            for j, x in enumerate(nbrs[i, 1:]):
                if i == x:
                    nbrs[i, 1:j+1] = nbrs[i, :j]
                    sims[i, 1:j+1] = sims[i, :j]
                    break
    sims = sims[:, 1:]
    nbrs = nbrs[:, 1:]

    x = [(np.array(nbr, dtype=np.uint32), np.array(sim, dtype=np.float32)) for nbr, sim in zip(nbrs, sims)]
    np.savez_compressed(knn_path, data=np.array(x))
    return nbrs, sims
