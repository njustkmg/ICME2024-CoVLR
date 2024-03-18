import numpy
import time
import numpy as np
import random

import os
import pickle
from torch import Tensor
import torch

from collections import OrderedDict
from math import log
import time
import warnings

warnings.filterwarnings("ignore")


# os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"

def cos_similar(p: Tensor, q: Tensor):
    print(p.shape, q.shape, type(p), type(q), q.transpose(0, -1).shape)
    sim_matrix = p.matmul(q.transpose(0, -1))
    a = torch.norm(p, dim=-1)
    b = torch.norm(q, dim=-1)
    # print("a , b : ", a.shape, b.shape)
    sim_matrix /= a.unsqueeze(-1)
    sim_matrix /= b.unsqueeze(-2)
    return np.array(sim_matrix)


def is_same(target, list):
    relation = []
    for i in range(len(list)):
        temp = 0
        for j in range(len(target)):
            if target[j] == 1 and list[i][j] == 1:
                temp = temp + 1
        relation.append(temp)
    return relation


def cos_similar_numpy(p, q):
    sim_matrix = np.zeros(len(q))
    for index in range(len(q)):
        sim_matrix[index] = float(np.dot(p, q[index])) / (np.linalg.norm(p) * np.linalg.norm(q[index]))
    return sim_matrix


def ndcg_i2i(embed_1, embed_2, relmatrix, sims, npts=None, threshold=500):
    if npts is None:
        npts = embed_1.shape[0]

    ndcgs = np.zeros(npts)
    for index in range(npts):
        im = embed_1[index].reshape(1, embed_1.shape[1])

        # Compute scores
        # d = np.dot(im, embed_2.T).flatten()
        # d = cos_similar_numpy(im, embed_2).flatten()
        d = sims[index]

        # 阻止自己检索自己
        d[index] = 0;
        relmatrix[index][index] = 0
        # get sims: d
        inds = np.argsort(d)[::-1]

        # compute NDCG
        inds_threshold = inds[0:threshold]  # 取前threshold个结果的索引
        if not np.all(relmatrix == 0):
            rel_threshold = relmatrix[index][inds_threshold]  # 取前threshold个结果的相关度
            rel_order_threshold = np.sort(relmatrix[index])[::-1][0:threshold]  # 所有结果排序后的相关度前threshold个
            dcg = 0.0
            idcg = 0.0
            for ind_t in range(threshold):
                dcg += rel_threshold[ind_t] / np.log2(ind_t + 2)
                idcg += rel_order_threshold[ind_t] / np.log2(ind_t + 2)
            if idcg > 0:
                ndcgs[index] = dcg / idcg
        # print("index 0 : ", index, inds_threshold, relmatrix[index][index * 5:(index+1) * 5] )
        # print("index 1 : ", np.around(rel_threshold, 3), np.around(rel_order_threshold, 3), round(ndcgs[index],3) )
    # Compute metrics
    ndcgs = ndcgs.mean()
    print("threshold : ", threshold, ndcgs)
    return ndcgs


def ndcg_t2t(embed_1, embed_2, relmatrix, sims, npts=None, threshold=500):
    if npts is None:
        npts = embed_1.shape[0]

    ndcgs = np.zeros(npts)
    for index in range(npts):
        im = embed_1[index].reshape(1, embed_1.shape[1])

        # Compute scores
        # d = np.dot(im, embed_2.T).flatten()
        # d = cos_similar_numpy(im, embed_2).flatten()
        d = sims[index]

        # 阻止自己检索自己
        # d[5*index : 5*(index+1)] = 0;relmatrix[index][5*index : 5*(index+1)] = 0
        d[5 * index] = 0;
        relmatrix[index][5 * index] = 0
        # get sims: d
        inds = np.argsort(d)[::-1]

        # compute NDCG
        inds_threshold = inds[0:threshold]  # 取前threshold个结果的索引
        if not np.all(relmatrix == 0):
            rel_threshold = relmatrix[index][inds_threshold]  # 取前threshold个结果的相关度
            rel_order_threshold = np.sort(relmatrix[index])[::-1][0:threshold]  # 所有结果排序后的相关度前threshold个
            dcg = 0.0
            idcg = 0.0
            for ind_t in range(threshold):
                dcg += rel_threshold[ind_t] / np.log2(ind_t + 2)
                idcg += rel_order_threshold[ind_t] / np.log2(ind_t + 2)
            if idcg > 0:
                ndcgs[index] = dcg / idcg
        # print("index 0 : ", index, inds_threshold, relmatrix[index][index * 5:(index+1) * 5] )
        # print("index 1 : ", np.around(rel_threshold, 3), np.around(rel_order_threshold, 3), round(ndcgs[index],3) )
        # print('\n')
    # Compute metrics
    ndcgs = ndcgs.mean()
    print("threshold : ", threshold, ndcgs)
    return ndcgs


def cal_ndcg_image(img_embs, image_label):
    start_time = time.time()
    relmatrix = np.dot(image_label, image_label.T)
    # relmatrix = np.memmap("./embed/coco-test-rougeL.npy", dtype=np.float32, mode='r+')
    relmatrix = np.load("./coco-rougeL.npy")
    # coco
    relmatrix = relmatrix.reshape(25000, -1)[:len(img_embs) * 5, :len(img_embs)][::5, :]
    # f30k
    #     relmatrix = relmatrix.reshape(5000, -1)[:len(img_embs)*5, :len(img_embs)][::5, :]
    # sims = np.dot(img_embs, img_embs.T)
    print("image shape : ", img_embs.shape, image_label.shape)
    print("relmatrix shape : ", relmatrix.shape)
    # sims = np.zeros((len(img_embs),len(img_embs)))
    # for index in range(len(sims)):
    #     sims[index] = cos_similar_numpy(img_embs[index], img_embs).flatten()

    sims = cos_similar(torch.tensor(img_embs).view(img_embs.shape[0], -1),
                       torch.tensor(img_embs).view(img_embs.shape[0], -1))

    ndcg_10 = ndcg_i2i(img_embs, img_embs, relmatrix, sims, npts=None, threshold=10)
    ndcg_20 = ndcg_i2i(img_embs, img_embs, relmatrix, sims, npts=None, threshold=20)
    ndcg_50 = ndcg_i2i(img_embs, img_embs, relmatrix, sims, npts=None, threshold=50)
    end_time = time.time()
    print("cost time : ", end_time - start_time)

    # start_time = time.time()
    # relmatrix = np.dot(image_label, image_label.T)
    # # relmatrix = np.memmap("./embed/coco-test-rougeL.npy", dtype=np.float32, mode='r+')
    # relmatrix = np.load("./coco-spice.npy")
    # # coco
    # relmatrix = relmatrix.reshape(25000, -1)[:len(img_embs) * 5, :len(img_embs)][::5, :]
    # # f30k
    # #     relmatrix = relmatrix.reshape(5000, -1)[:len(img_embs)*5, :len(img_embs)][::5, :]
    # # sims = np.dot(img_embs, img_embs.T)
    # print("image shape : ", img_embs.shape, image_label.shape)
    # print("relmatrix shape : ", relmatrix.shape)
    # # sims = np.zeros((len(img_embs),len(img_embs)))
    # # for index in range(len(sims)):
    # #     sims[index] = cos_similar_numpy(img_embs[index], img_embs).flatten()
    #
    # sims = cos_similar(torch.tensor(img_embs).view(img_embs.shape[0], -1),
    #                    torch.tensor(img_embs).view(img_embs.shape[0], -1))
    #
    # ndcg_10 = ndcg_i2i(img_embs, img_embs, relmatrix, sims, npts=None, threshold=10)
    # ndcg_20 = ndcg_i2i(img_embs, img_embs, relmatrix, sims, npts=None, threshold=20)
    # ndcg_50 = ndcg_i2i(img_embs, img_embs, relmatrix, sims, npts=None, threshold=50)
    # end_time = time.time()
    # print("cost time : ", end_time - start_time)
    return ndcg_10


def cal_ndcg_test(cap_embs, cap_label):
    start_time = time.time()
    relmatrix = np.dot(cap_label, cap_label.T)
    # relmatrix = np.memmap("./embed/coco-test-rougeL.npy", dtype=np.float32, mode='r+')
    relmatrix = np.load("./coco-rougeL.npy")
    # coco
    relmatrix = relmatrix.reshape(25000, -1)[:len(cap_embs), :len(cap_embs) // 5]
    # f30k
    #     relmatrix = relmatrix.reshape(5000, -1)[:len(cap_embs), :len(cap_embs)//5]
    relmatrix = relmatrix.T
    query = cap_embs[::5, :]
    # sims = np.dot(img_embs, img_embs.T)
    print("cap shape : ", query.shape, cap_embs.shape, cap_label.shape)
    print("relmatrix shape : ", relmatrix.shape)
    '''sims = np.zeros((len(query),len(cap_embs)))
    for index in range(len(sims)):
        sims[index] = cos_similar_numpy(query[index], cap_embs).flatten()'''

    sims = cos_similar(torch.tensor(cap_embs).view(cap_embs.shape[0], -1),
                       torch.tensor(cap_embs).view(cap_embs.shape[0], -1))
    sims = sims[::5, :]
    print("sims shape : ", sims.shape)

    ndcg_10 = ndcg_t2t(query, cap_embs, relmatrix, sims, npts=None, threshold=10)
    ndcg_20 = ndcg_t2t(query, cap_embs, relmatrix, sims, npts=None, threshold=20)
    ndcg_50 = ndcg_t2t(query, cap_embs, relmatrix, sims, npts=None, threshold=50)
    end_time = time.time()
    print("cost time : ", end_time - start_time)

    # start_time = time.time()
    # relmatrix = np.dot(cap_label, cap_label.T)
    # # relmatrix = np.memmap("./embed/coco-test-rougeL.npy", dtype=np.float32, mode='r+')
    # relmatrix = np.load("./coco-spice.npy")
    # # coco
    # relmatrix = relmatrix.reshape(25000, -1)[:len(cap_embs), :len(cap_embs) // 5]
    # # f30k
    # #     relmatrix = relmatrix.reshape(5000, -1)[:len(cap_embs), :len(cap_embs)//5]
    # relmatrix = relmatrix.T
    # query = cap_embs[::5, :]
    # # sims = np.dot(img_embs, img_embs.T)
    # print("cap shape : ", query.shape, cap_embs.shape, cap_label.shape)
    # print("relmatrix shape : ", relmatrix.shape)
    # '''sims = np.zeros((len(query),len(cap_embs)))
    # for index in range(len(sims)):
    #     sims[index] = cos_similar_numpy(query[index], cap_embs).flatten()'''
    #
    # sims = cos_similar(torch.tensor(cap_embs).view(cap_embs.shape[0], -1),
    #                    torch.tensor(cap_embs).view(cap_embs.shape[0], -1))
    # sims = sims[::5, :]
    # print("sims shape : ", sims.shape)
    #
    # ndcg_10 = ndcg_t2t(query, cap_embs, relmatrix, sims, npts=None, threshold=10)
    # ndcg_20 = ndcg_t2t(query, cap_embs, relmatrix, sims, npts=None, threshold=20)
    # ndcg_50 = ndcg_t2t(query, cap_embs, relmatrix, sims, npts=None, threshold=50)
    # end_time = time.time()
    # print("cost time : ", end_time - start_time)
    return ndcg_10


def i2t(images, captions, caplens, sims, npts=None, return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    npts = images.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            # print(np.where(inds == i))
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, captions, caplens, sims, npts=None, return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    npts = images.shape[0]
    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)

    # --> (5N(caption), N(image))
    # sims = sims.T

    for index in range(npts):
        for i in range(5):
            inds = np.argsort(sims[5 * index + i])[::-1]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def CMR_mertic(img_embs, cap_embs, cap_lens, sims_i2t, sims_t2i):
    print("embed shape : ", img_embs.shape, cap_embs.shape, len(cap_lens))
    # if sims == None:
    #     sims = np.zeros((len(img_embs),len(cap_embs)))
    #     for index in range(len(sims)):
    #         sims[index] = cos_similar_numpy(img_embs[index], cap_embs).flatten()

    r, rt = i2t(img_embs, cap_embs, cap_lens, sims_i2t, return_ranks=True)
    ri, rti = t2i(img_embs, cap_embs, cap_lens, sims_t2i, return_ranks=True)
    ar = (r[0] + r[1] + r[2]) / 3
    ari = (ri[0] + ri[1] + ri[2]) / 3
    rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
    print("rsum: %.1f" % rsum)
    print("Average i2t Recall: %.1f" % ar)
    print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
    print("Average t2i Recall: %.1f" % ari)
    print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)


def CMR_fusion_mertic(img_embs, cap_embs, cap_lens):
    print("embed shape : ", img_embs.shape, cap_embs.shape, len(cap_lens))
    sims = np.zeros((len(img_embs), len(cap_embs)))

    sims = cos_similar(torch.tensor(img_embs).view(img_embs.shape[0], -1),
                       torch.tensor(cap_embs).view(cap_embs.shape[0], -1))

    r, rt = i2t(img_embs, cap_embs, cap_lens, sims, return_ranks=True)
    ri, rti = t2i(img_embs, cap_embs, cap_lens, sims.T, return_ranks=True)
    ar = (r[0] + r[1] + r[2]) / 3
    ari = (ri[0] + ri[1] + ri[2]) / 3
    rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
    print("rsum: %.1f" % rsum)
    print("Average i2t Recall: %.1f" % ar)
    print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
    print("Average t2i Recall: %.1f" % ari)
    print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)


if __name__ == '__main__':

    path = 'coco5k_dis'

    cap_embs = np.load(f'{path}/text_embed.npy')  # [::128,:]
    cap_label = np.load(f'{path}/text_embed.npy')  # [::128,:]
    print(cap_embs.shape)
    cal_ndcg_test(cap_embs, cap_label)
    image_embed = np.load(f'{path}/image_embed.npy', allow_pickle=True)
    print("image_embed : ", image_embed.shape)
    if image_embed.shape[-1] == 1024:
        image_embed = image_embed.reshape(-1, 145, 1024)
    else:
        image_embed = image_embed.reshape(-1, 577, 768)
    print("image_embed : ", image_embed.shape)
    #     image_embed = np.mean(image_embed, axis=1)
    image_embed = image_embed[:, 0, :]
    image_label = np.load(f'{path}/image_embed.npy', allow_pickle=True)
    image_label = image_embed
    print("image_embed : ", image_embed.shape)
    cal_ndcg_image(image_embed, image_label)
