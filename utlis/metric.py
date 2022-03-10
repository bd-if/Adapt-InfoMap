#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
from sklearn.metrics.cluster import (contingency_matrix, normalized_mutual_info_score)
from sklearn.metrics import (precision_score, recall_score)

__all__ = ['pairwise', 'bcubed', 'nmi', 'precision', 'recall', 'accuracy', 'new_metrics']


def _check(gt_labels, pred_labels):
    if gt_labels.ndim != 1:
        raise ValueError("gt_labels must be 1D: shape is %r" %
                         (gt_labels.shape, ))
    if pred_labels.ndim != 1:
        raise ValueError("pred_labels must be 1D: shape is %r" %
                         (pred_labels.shape, ))
    if gt_labels.shape != pred_labels.shape:
        raise ValueError(
            "gt_labels and pred_labels must have same size, got %d and %d" %
            (gt_labels.shape[0], pred_labels.shape[0]))
    return gt_labels, pred_labels


def _get_lb2idxs(labels):
    lb2idxs = {}
    for idx, lb in enumerate(labels):
        if lb not in lb2idxs:
            lb2idxs[lb] = []
        lb2idxs[lb].append(idx)
    return lb2idxs


def _compute_fscore(pre, rec):
    return 2. * pre * rec / (pre + rec)


def fowlkes_mallows_score(gt_labels, pred_labels, sparse=True):

    n_samples, = gt_labels.shape

    c = contingency_matrix(gt_labels, pred_labels, sparse=sparse)
    tk = np.dot(c.data, c.data) - n_samples
    pk = np.sum(np.asarray(c.sum(axis=0)).ravel()**2) - n_samples
    qk = np.sum(np.asarray(c.sum(axis=1)).ravel()**2) - n_samples
    avg_pre = tk / pk
    avg_rec = tk / qk
    fscore = _compute_fscore(avg_pre, avg_rec)
    return avg_pre, avg_rec, fscore


def pairwise(gt_labels, pred_labels, sparse=True):
    _check(gt_labels, pred_labels)
    avg_pre, avg_rec, fscore = fowlkes_mallows_score(gt_labels, pred_labels, sparse)
    print('pairwise: avg_pre:{:.4f}, avg_rec:{:.4f}, fscore:{:.4f}'.format(avg_pre, avg_rec, fscore))
    return avg_pre, avg_rec, fscore


def bcubed(gt_labels, pred_labels):
    _check(gt_labels, pred_labels)

    gt_lb2idxs = _get_lb2idxs(gt_labels)
    pred_lb2idxs = _get_lb2idxs(pred_labels)

    num_lbs = len(gt_lb2idxs)
    pre = np.zeros(num_lbs)
    rec = np.zeros(num_lbs)
    gt_num = np.zeros(num_lbs)

    s1 = 0
    for i, gt_idxs in enumerate(gt_lb2idxs.values()):
        all_pred_lbs = np.unique(pred_labels[gt_idxs])
        gt_num[i] = len(gt_idxs)
        for pred_lb in all_pred_lbs:
            pred_idxs = pred_lb2idxs[pred_lb]
            n = 1. * np.intersect1d(gt_idxs, pred_idxs).size
            if len(pred_idxs) > 1:
                pass
            else:
                s1 += 1

            pre[i] += n ** 2 / len(pred_idxs)
            rec[i] += n**2 / gt_num[i]
    # print('singleton:', s1)

    gt_num = gt_num.sum()
    avg_pre = pre.sum() / gt_num
    avg_rec = rec.sum() / gt_num
    fscore = _compute_fscore(avg_pre, avg_rec)
    print('bcubed: avg_pre:{:.4f}, avg_rec:{:.4f}, fscore:{:.4f}'.format(avg_pre, avg_rec, fscore))
    return avg_pre, avg_rec, fscore


def new_metrics(gt_labels, pred_labels, sparse=True):

    c = contingency_matrix(gt_labels, pred_labels, sparse=sparse)

    true_class = np.asarray(c.sum(axis=1)).ravel()
    pred_cluster = np.asarray(c.sum(axis=0)).ravel()
    true_class_num = true_class.shape[0]
    pred_cluster_num = pred_cluster.shape[0]

    print('R#s: {:.4f}'.format(np.sum(c.sum(axis=0) <= 1) / true_class_num))
    print('R#i: {}/{}={:.4f}'.format(pred_cluster_num, true_class_num, pred_cluster_num/true_class_num))

    IDTP1, IDTP2, IDTP3, IDTP4, IDTP5 = 0,0,0,0,0

    for j in range(c.shape[1]):
        x = c[:, j]
        pre = x.data / pred_cluster[j]
        rec = x.toarray().ravel() / true_class
        rec = rec[rec > 0.0]

        IDTP1 += np.sum((pre > 0.5) & (rec > 0.5))
        IDTP2 += np.sum((pre > 0.6) & (rec > 0.6))
        IDTP3 += np.sum((pre > 0.7) & (rec > 0.7))
        IDTP4 += np.sum((pre > 0.8) & (rec > 0.8))
        IDTP5 += np.sum((pre > 0.9) & (rec > 0.9))

    IDFP1 = pred_cluster_num - IDTP1
    IDFP2 = pred_cluster_num - IDTP2
    IDFP3 = pred_cluster_num - IDTP3
    IDFP4 = pred_cluster_num - IDTP4
    IDFP5 = pred_cluster_num - IDTP5
    IDFN1 = true_class_num - IDTP1
    IDFN2 = true_class_num - IDTP2
    IDFN3 = true_class_num - IDTP3
    IDFN4 = true_class_num - IDTP4
    IDFN5 = true_class_num - IDTP5
    IDF11 = IDTP1/(IDTP1+IDFP1/2+IDFN1/2)
    IDF12 = IDTP2/(IDTP2+IDFP2/2+IDFN2/2)
    IDF13 = IDTP3/(IDTP3+IDFP3/2+IDFN3/2)
    IDF14 = IDTP4/(IDTP4+IDFP4/2+IDFN4/2)
    IDF15 = IDTP5/(IDTP5+IDFP5/2+IDFN5/2)

    print('IDTF: (0.5):{}, (0.6):{}, (0.7):{}, (0.8):{}, (0.9):{}'.format(IDTP1,IDTP2,IDTP3,IDTP4,IDTP5))
    print('IDF1: (0.5):{:.4f}, (0.6):{:.4f}, (0.7):{:.4f}, (0.8):{:.4f}, (0.9):{:.4f}'
          .format(IDF11,IDF12,IDF13,IDF14,IDF15))


def nmi(gt_labels, pred_labels):
    return normalized_mutual_info_score(pred_labels, gt_labels)


def precision(gt_labels, pred_labels):
    return precision_score(gt_labels, pred_labels)


def recall(gt_labels, pred_labels):
    return recall_score(gt_labels, pred_labels)


def accuracy(gt_labels, pred_labels):
    return np.mean(gt_labels == pred_labels)
