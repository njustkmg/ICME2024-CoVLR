from __future__ import print_function
import numpy

import time
import numpy as np
import torch
from torch import Tensor
from math import log
import tqdm
from collections import OrderedDict


# from utils import dot_sim, get_model
# from evaluate_utils.dcg import DCG
# from models.loss import order_sim, AlignmentContrastiveLoss

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cos_similar(p: Tensor, q: Tensor):
    sim_matrix = p.matmul(q.transpose(-2, -1))
    a = torch.norm(p, p=2, dim=-1)
    b = torch.norm(q, p=2, dim=-1)
    sim_matrix /= a.unsqueeze(-1)
    sim_matrix /= b.unsqueeze(-2)
    return sim_matrix


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
        #         d[index] = 0;relmatrix[index][index] = 0
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
        #         d[5*index] = 0;relmatrix[index][5*index] = 0
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


def cal_ndcg_image(img_embs, cap_embs, image_label):
    start_time = time.time()
    relmatrix_text = np.load("vizwiz-rougeL.npy")
    relmatrix_text = relmatrix_text.reshape(5000, -1)[:len(cap_embs), :len(cap_embs) // 5].T
    # relmatrix_text = relmatrix_text
    query = cap_embs[::5, :]
    print("cap shape : ", query.shape, cap_embs.shape)
    print("relmatrix_text shape : ", relmatrix_text.shape)
    sims_text = cos_similar(torch.from_numpy(query), torch.from_numpy(cap_embs)).numpy()
    sims_text = np.load('vizwiz/score_matrix_i2t.npy')
    print('sims_text', sims_text.shape)

    # tupian
    relmatrix_image = np.load("vizwiz-rougeL.npy")
    relmatrix_image = relmatrix_image.reshape(5000, -1)[:len(img_embs) * 5, :len(img_embs)][::5, :]
    sims_image = cos_similar(torch.from_numpy(img_embs), torch.from_numpy(img_embs)).numpy()
    print("relmatrix_image shape : ", relmatrix_image.shape)
    print('sims_image', sims_image.shape, sims_text.shape)
    relmatrix = np.concatenate((relmatrix_image, relmatrix_text), axis=1)
    sims = np.concatenate((sims_image, sims_text), axis=1)
    print('relmatrix', relmatrix.shape)
    print('sims', sims.shape)

    ndcgi2i = ndcg_i2i(img_embs, img_embs, relmatrix, sims, npts=None, threshold=10)
    ndcgi2i = ndcg_i2i(img_embs, img_embs, relmatrix, sims, npts=None, threshold=20)
    ndcgi2i = ndcg_i2i(img_embs, img_embs, relmatrix, sims, npts=None, threshold=50)
    end_time = time.time()
    print("cost time : ", end_time - start_time)

    # start_time = time.time()
    # relmatrix_text = np.load("coco-spice.npy")
    # relmatrix_text = relmatrix_text.reshape(25000, -1)[:len(cap_embs), :len(cap_embs)//5].T
    # # relmatrix_text = relmatrix_text
    # query = cap_embs[::5, :]
    # print("cap shape : ", query.shape, cap_embs.shape)
    # print("relmatrix_text shape : ", relmatrix_text.shape)
    # sims_text = cos_similar(torch.from_numpy(query), torch.from_numpy(cap_embs)).numpy()
    # sims_text = np.load('coco1k/score_matrix_i2t.npy')
    # print('sims_text', sims_text.shape)
    #
    # # tupian
    # relmatrix_image = np.load("coco-spice.npy")
    # relmatrix_image = relmatrix_image.reshape(25000, -1)[:len(img_embs)*5, :len(img_embs)][::5, :]
    # sims_image = cos_similar(torch.from_numpy(img_embs), torch.from_numpy(img_embs)).numpy()
    # print("relmatrix_image shape : ", relmatrix_image.shape)
    # print('sims_image', sims_image.shape, sims_text.shape)
    # relmatrix = np.concatenate((relmatrix_image, relmatrix_text),axis=1)
    # sims = np.concatenate((sims_image, sims_text),axis=1)
    # print('relmatrix',relmatrix.shape)
    # print('sims',sims.shape)
    #
    # ndcgi2i = ndcg_i2i(img_embs, img_embs, relmatrix, sims, npts=None, threshold=10)
    # ndcgi2i = ndcg_i2i(img_embs, img_embs, relmatrix, sims, npts=None, threshold=20)
    # ndcgi2i = ndcg_i2i(img_embs, img_embs, relmatrix, sims, npts=None, threshold=50)
    # end_time = time.time()
    # print("cost time : ", end_time - start_time)


def cal_ndcg_text(cap_embs, img_embs, cap_label):
    start_time = time.time()
    # relmatrix = np.dot(cap_label, cap_label.T)
    relmatrix_text = np.load("vizwiz-rougeL.npy")
    relmatrix_text = relmatrix_text.reshape(5000, -1)[:len(cap_embs), :len(cap_embs) // 5]
    relmatrix_text = relmatrix_text.T
    query = cap_embs[::5, :]
    print("cap shape : ", query.shape, cap_embs.shape, cap_label.shape)
    print("relmatrix shape : ", relmatrix_text.shape)
    sims_text = cos_similar(torch.from_numpy(query), torch.from_numpy(cap_embs)).numpy()
    print('sims_text', sims_text.shape)

    # tupian
    relmatrix_image = np.load("vizwiz-rougeL.npy")
    relmatrix_image = relmatrix_image.reshape(5000, -1)[:len(img_embs) * 5, :len(img_embs)][::5, :].T
    sims_image = cos_similar(torch.from_numpy(img_embs), torch.from_numpy(img_embs)).numpy()
    sims_image = np.load('vizwiz/score_matrix_t2i.npy').T[:, ::5]
    print("relmatrix_image shape : ", relmatrix_image.shape)
    print('sims_image', sims_image.shape)

    relmatrix = np.concatenate((relmatrix_text, relmatrix_image), axis=1)
    sims = np.concatenate((sims_text, sims_image), axis=1)
    print('relmatrix', relmatrix.shape)
    print('sims', sims.shape)

    ndcgi2i = ndcg_t2t(query, cap_embs, relmatrix, sims, npts=None, threshold=10)
    ndcgi2i = ndcg_t2t(query, cap_embs, relmatrix, sims, npts=None, threshold=20)
    ndcgi2i = ndcg_t2t(query, cap_embs, relmatrix, sims, npts=None, threshold=50)
    end_time = time.time()
    print("cost time : ", end_time - start_time)

    # start_time = time.time()
    # # relmatrix = np.dot(cap_label, cap_label.T)
    # relmatrix_text = np.load("coco-spice.npy")
    # relmatrix_text = relmatrix_text.reshape(25000, -1)[:len(cap_embs), :len(cap_embs)//5]
    # relmatrix_text = relmatrix_text.T
    # query = cap_embs[::5, :]
    # print("cap shape : ", query.shape, cap_embs.shape, cap_label.shape)
    # print("relmatrix shape : ", relmatrix_text.shape)
    # sims_text = cos_similar(torch.from_numpy(query), torch.from_numpy(cap_embs)).numpy()
    # print('sims_text',sims_text.shape)
    #
    # #tupian
    # relmatrix_image = np.load("coco-spice.npy")
    # relmatrix_image = relmatrix_image.reshape(25000, -1)[:len(img_embs)*5, :len(img_embs)][::5, :].T
    # sims_image = cos_similar(torch.from_numpy(img_embs), torch.from_numpy(img_embs)).numpy()
    # sims_image = np.load('coco1k/score_matrix_t2i.npy').T[:,::5]
    # print("relmatrix_image shape : ", relmatrix_image.shape)
    # print('sims_image',sims_image.shape )
    #
    # relmatrix = np.concatenate((relmatrix_text,relmatrix_image),axis=1)
    # sims = np.concatenate((sims_text,sims_image),axis=1)
    # print('relmatrix',relmatrix.shape)
    # print('sims',sims.shape)
    #
    # ndcgi2i = ndcg_t2t(query, cap_embs, relmatrix, sims, npts=None, threshold=10)
    # ndcgi2i = ndcg_t2t(query, cap_embs, relmatrix, sims, npts=None, threshold=20)
    # ndcgi2i = ndcg_t2t(query, cap_embs, relmatrix, sims, npts=None, threshold=50)
    # end_time = time.time()
    # print("cost time : ", end_time - start_time)


if __name__ == '__main__':

    # methods = "vsepp"
    # cap_embs = np.load('../' + methods + '/embed/cap_embs_f30k.npy')#[::128,:]
    # cap_label = np.load('../' + methods + '/embed/cap_embs_f30k.npy')#[::128,:]
    # print(cap_embs.shape, len(cap_embs.shape))
    # if len(cap_embs.shape) == 3:
    #     cap_embs = np.mean(cap_embs,axis=1)
    #     cap_label = np.mean(cap_label,axis=1)
    # print(cap_embs.shape)
    # image_embed = np.load('../' + methods + '/embed/img_embs_f30k.npy',allow_pickle=True)
    # image_label = np.load('../' + methods + '/embed/img_embs_f30k.npy',allow_pickle=True)
    # print("image_embed : ", image_embed.shape)
    # if len(image_embed.shape) == 3:
    #     image_embed = np.mean(image_embed,axis=1)
    #     image_label = np.mean(image_label,axis=1)
    # image_embed = image_embed[::5,:]
    # image_label = image_label[::5,:]
    # image_label = image_embed
    # print("image_embed : ", image_embed.shape)
    # cal_ndcg_text(cap_embs, image_embed, cap_label)
    # cal_ndcg_image(image_embed, cap_embs, image_label)

    path = 'vizwiz'

    cap_embs = np.load(f'./{path}/text_embed.npy')  # [::128,:]
    cap_label = np.load(f'./{path}/text_embed.npy')  # [::128,:]
    print(cap_embs.shape, len(cap_embs.shape))
    if len(cap_embs.shape) == 3:
        cap_embs = np.mean(cap_embs, axis=1)
        cap_label = np.mean(cap_label, axis=1)
    print(cap_embs.shape)
    image_embed = np.load(f'./{path}/image_embed.npy', allow_pickle=True)
    image_label = np.load(f'./{path}/image_embed.npy', allow_pickle=True)
    print("image_embed : ", image_embed.shape)
    # if len(image_embed.shape) == 3:
    #     image_embed = np.mean(image_embed,axis=1)
    #     image_label = np.mean(image_label,axis=1)
    if image_embed.shape[-1] == 1024:
        image_embed = image_embed.reshape(-1, 145, 1024)
    else:
        image_embed = image_embed.reshape(-1, 577, 768)
    image_embed = image_embed[:, 0, :]
    image_label = image_embed
    print("image_embed : ", image_embed.shape)
    cal_ndcg_text(cap_embs, image_embed, cap_label)
    cal_ndcg_image(image_embed, cap_embs, image_label)
