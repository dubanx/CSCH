import logging
import json
import numpy as np
import torch
import os
import configs
from utils.misc import Timer
from tqdm import tqdm

def get_hamm_dist(codes, centroids, margin=0, normalize=False):
    with torch.no_grad():
        nbit = centroids.size(1)
        dist = 0.5 * (nbit - torch.matmul(codes.sign(), centroids.sign().t()))

        if normalize:
            dist = dist / nbit

        if margin == 0:
            return dist
        else:
            codes_clone = codes.clone()
            codes_clone[codes_clone.abs() < margin] = 0
            dist_margin = 0.5 * (nbit - torch.matmul(codes_clone.sign(), centroids.sign().t()))
            if normalize:
                dist_margin = dist_margin / nbit
            return dist, dist_margin


def get_codes_and_labels(model, loader):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    vs = []
    ts = []
    for e, (d, t) in enumerate(loader):
        print(f'[{e + 1}/{len(loader)}]', end='\r')
        with torch.no_grad():
            # model forward
            d, t = d.to(device), t.to(device)
            v = model(d)
            if isinstance(v, tuple):
                v = v[0]

            vs.append(v)
            ts.append(t)

    print()
    vs = torch.cat(vs)
    ts = torch.cat(ts)
    return vs, ts

def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH

def CalcTopMap(rB, qB, retrievalL, queryL, topk):
    rB = rB.clone()
    qB = qB.clone()
    rB = torch.sign(rB).cpu().numpy()  # (ndb, nbit)
    qB = torch.sign(qB).cpu().numpy()  # (nq, nbit)
    retrievalL = retrievalL.cpu().numpy()
    queryL = queryL.cpu().numpy()

    num_query = queryL.shape[0]
    topkmap = 0
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        hamm_ = torch.from_numpy(hamm)
        # print(hamm_.shape)
        ind = torch.topk(hamm_, topk, dim=0, largest=False)[1].cpu().numpy()
        # ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap

def CalcTopMapWithPR(qB, queryL, rB, retrievalL, topk):
    rB = rB.clone()
    qB = qB.clone()
    rB = torch.sign(rB).cpu().numpy()  # (ndb, nbit)
    qB = torch.sign(qB).cpu().numpy()  # (nq, nbit)
    retrievalL = retrievalL.cpu().numpy()
    queryL = queryL.cpu().numpy()

    num_query = queryL.shape[0]
    num_gallery = retrievalL.shape[0]
    topkmap = 0
    prec = np.zeros((num_query, num_gallery))
    recall = np.zeros((num_query, num_gallery))
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        # hamm_ = torch.from_numpy(hamm)
        # ind = torch.topk(hamm_, topk, dim=0, largest=False)[1].cpu().numpy()
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)
        all_sim_num = np.sum(gnd)

        prec_sum = np.cumsum(gnd)
        return_images = np.arange(1, num_gallery + 1)

        prec[iter, :] = prec_sum / return_images
        recall[iter, :] = prec_sum / all_sim_num

        assert recall[iter, -1] == 1.0
        assert all_sim_num == prec_sum[-1]

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    index = np.argwhere(recall[:, -1] == 1.0)
    index = index.squeeze()
    prec = prec[index]
    recall = recall[index]
    cum_prec = np.mean(prec, 0)
    cum_recall = np.mean(recall, 0)

    return topkmap, cum_prec, cum_recall

def precison_topK(config, qB, queryL, rB, retrievalL, topk=1000):
    rB = rB.clone()
    qB = qB.clone()
    rB = torch.sign(rB).cpu().numpy()  # (ndb, nbit)
    qB = torch.sign(qB).cpu().numpy()  # (nq, nbit)
    retrievalL = retrievalL.cpu().numpy()
    queryL = queryL.cpu().numpy()

    num_query = queryL.shape[0]
    num_gallery = retrievalL.shape[0]
    topkmap = 0
    prec = np.zeros((num_query, num_gallery))
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        # hamm_ = torch.from_numpy(hamm)
        # ind = torch.topk(hamm_, topk, dim=0, largest=False)[1].cpu().numpy()
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)
        all_sim_num = np.sum(gnd)

        prec_sum = np.cumsum(gnd)
        return_images = np.arange(1, num_gallery + 1)

        prec[iter, :] = prec_sum / return_images


        assert all_sim_num == prec_sum[-1]

    cum_prec = np.mean(prec, 0)
    cum_prec = cum_prec[0:topk]
    index = [i for i in range(1, topk+1)]
    p_data = {
        "index": index,
        "P": cum_prec.tolist(),
    }
    os.makedirs(os.path.dirname(config["pr_curve_path"]), exist_ok=True)
    with open(config["pr_curve_path"], 'w') as f:
        f.write(json.dumps(p_data))
    print(cum_prec.shape)
    print(cum_prec[0])

    return cum_prec

def pr(config, tst_binary, tst_label, trn_binary, trn_label, num_dataset):
    mAP, cum_prec, cum_recall = CalcTopMapWithPR(tst_binary, tst_label,
                                                     trn_binary, trn_label,
                                                     config['R'])
    # num_dataset = 1000
    # tmp = num_dataset // 11
    index_range = num_dataset // 10
    index = [i * 10 - 1 for i in range(1, index_range + 1)]
    max_index = max(index)
    overflow = num_dataset - index_range * 10
    index = index + [max_index + i for i in range(1, overflow + 1)]
    c_prec = cum_prec[index]
    c_recall = cum_recall[index]

    pr_data = {
        "index": index,
        "P": c_prec.tolist(),
        "R": c_recall.tolist()
    }
    os.makedirs(os.path.dirname(config["pr_curve_path"]), exist_ok=True)
    with open(config["pr_curve_path"], 'w') as f:
        f.write(json.dumps(pr_data))
    print("mAP:", mAP)
    print("pr curve save to ", config["pr_curve_path"])

def precision_recall(config, query_code,
                           query_targets,
                           retrieval_code,
                           retrieval_targets,
                           ):
    """
    Calculate mean average precision(map).

    Args:
        query_code (torch.Tensor): Query data hash code.
        retrieval_code (torch.Tensor): Database data hash code.
        query_targets (torch.Tensor): Query data targets, one-hot
        retrieval_targets (torch.Tensor): Database data targets, one-host
        device (torch.device): Using CPU or GPU.
        topk (int): Calculate top k data map. 

    Returns:
        meanAP (float): Mean Average Precision.
    """
    num_query = query_targets.shape[0]
    device = torch.device(config.get('device', 'cuda:1'))
    # mean_AP = 0.0
    p_r = torch.zeros(11).to(device)
    for i in range(num_query):
        # Retrieve images from database
        retrieval = (query_targets[i, :] @ retrieval_targets.t() > 0).float()

        # Calculate hamming distance
        hamming_dist = 0.5 * (retrieval_code.shape[1] - query_code[i, :] @ retrieval_code.t())

        # Arrange position according to hamming distance
        topk = config['R']
        retrieval = retrieval[torch.argsort(hamming_dist)][:topk]

        # Retrieval count
        retrieval_cnt = retrieval.sum().int().item()
        # import ipdb;ipdb.set_trace()
        # Can not retrieve images
        if retrieval_cnt == 0:
            continue

        # Generate score for every position
        score = torch.linspace(1, retrieval_cnt, retrieval_cnt).to(device)

        # Acquire index
        index = (torch.nonzero(retrieval == 1).squeeze() + 1.0).float()# 返回按照相似度排序后的retrieval中正例的index

        # import ipdb;ipdb.set_trace()
        temp =  (score / index)
        sampled_index = torch.linspace(0, retrieval_cnt-1, 11).int().to(device)
        # import ipdb;ipdb.set_trace()
        sampled_PR = temp.index_select(0, sampled_index)
        p_r += sampled_PR
        # import ipdb;ipdb.set_trace()
        # mean_AP += (score / index).mean() # 表示找到第score个正例时的准确率
    return p_r / num_query

def calculate_mAP(db_codes, db_labels,
                  test_codes, test_labels,
                  R, threshold=0.):
    # clone in case changing value of the original codes
    db_codes = db_codes.clone()
    test_codes = test_codes.clone()

    # if value within margin, set to 0
    if threshold != 0:
        db_codes[db_codes.abs() < threshold] = 0
        test_codes[test_codes.abs() < threshold] = 0

    # binarized
    db_codes = torch.sign(db_codes)  # (ndb, nbit)
    test_codes = torch.sign(test_codes)  # (nq, nbit)

    db_labels = db_labels.cpu().numpy()
    test_labels = test_labels.cpu().numpy()

    dist = []
    nbit = db_codes.size(1)

    timer = Timer()
    total_timer = Timer()

    timer.tick()
    total_timer.tick() 

    with torch.no_grad():
        db_codes_ttd = configs.tensor_to_dataset(db_codes)
        db_codes_loader = configs.dataloader(db_codes_ttd, 32, False, 0, False)

        # calculate hamming distance
        for i, db_code in enumerate(db_codes_loader):
            dist.append(0.5 * (nbit - torch.matmul(test_codes, db_code.t())).cpu())
            timer.toc()
            print(f'Distance [{i + 1}/{len(db_codes_loader)}] ({timer.total:.2f}s)', end='\r')

        dist = torch.cat(dist, 1)  # .numpy()
        print(dist.shape)

    # fast sort
    timer.tick()
    # different sorting will have affect on mAP score! because the order with same hamming distance might be diff.
    # unsorted_ids = np.argpartition(dist, R - 1)[:, :R]

    # torch sorting is quite fast, pytorch ftw!!!
    topk_ids = torch.topk(dist, R, dim=1, largest=False)[1].cpu()
    timer.toc()
    print(f'Sorting ({timer.total:.2f}s)')

    # calculate mAP
    timer.tick()
    APx = []
    for i in range(dist.shape[0]):
        label = test_labels[i, :]
        label[label == 0] = -1
        idx = topk_ids[i, :]
        # idx = idx[np.argsort(dist[i, :][idx])]
        imatch = np.sum(np.equal(db_labels[idx[0: R], :], label), 1) > 0
        rel = np.sum(imatch)
        Lx = np.cumsum(imatch)
        Px = Lx.astype(float) / np.arange(1, R + 1, 1)
        if rel != 0:
            APx.append(np.sum(Px * imatch) / rel)
        else:  # didn't retrieve anything relevant
            APx.append(0)
        timer.toc()
        print(f'Query [{i + 1}/{dist.shape[0]}] ({timer.total:.2f}s)', end='\r')

    print()
    total_timer.toc()
    logging.info(f'Total time usage for calculating mAP: {total_timer.total:.2f}s')

    return np.mean(np.array(APx))


def sign_dist(inputs, centroids, margin=0):
    n, b1 = inputs.size()
    nclass, b2 = centroids.size()

    assert b1 == b2, 'inputs and centroids must have same number of bit'

    # sl = relu(margin - x*y)
    out = inputs.view(n, 1, b1) * centroids.sign().view(1, nclass, b1)
    out = torch.relu(margin - out)  # (n, nclass, nbit)

    return out


def calculate_similarity_matrix(centroids):
    nclass = centroids.size(0)
    sim = torch.zeros(nclass, nclass, device=centroids.device)

    for rc in range(nclass):
        for cc in range(nclass):
            sim[rc, cc] = (centroids[rc] == centroids[cc]).float().mean()

    return sim
