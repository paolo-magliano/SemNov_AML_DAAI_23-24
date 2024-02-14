import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.sncore_4k import *
# noinspection PyUnresolvedReferences
from datasets.sncore_splits import *
from utils.utils import *
from utils.dist import *
from sklearn import metrics as skm
from utils.ood_metrics import calc_metrics
from tqdm import tqdm
import collections

try:
    # flag for disabling TQDM during evaluation!
    DISABLE_TQDM = bool(os.environ['NO_TQDM'])
except KeyError:
    DISABLE_TQDM = False


def print_ood_output(res_tar1, res_tar2, res_big_tar):
    # invert aupr in and out as we use label 1 for ID data
    auroc1, fpr1, auprin1, auprout1, f1_threshold1, f1_accuracy1, j_threshold1, j_accuracy1 = res_tar1['auroc'], res_tar1['fpr_at_95_tpr'], res_tar1['aupr_in'], res_tar1['aupr_out'], res_tar1['f1_threshold'], res_tar1['f1_accuracy'], res_tar1['j_threshold'], res_tar1['j_accuracy']
    auroc2, fpr2, auprin2, auprout2, f1_threshold2, f1_accuracy2, j_threshold2, j_accuracy2 = res_tar2['auroc'], res_tar2['fpr_at_95_tpr'], res_tar2['aupr_in'], res_tar2['aupr_out'], res_tar2['f1_threshold'], res_tar2['f1_accuracy'], res_tar2['j_threshold'], res_tar2['j_accuracy']
    auroc3, fpr3, auprin3, auprout3, f1_threshold3, f1_accuracy3, j_threshold3, j_accuracy3 = res_big_tar['auroc'], res_big_tar['fpr_at_95_tpr'], res_big_tar['aupr_in'], res_big_tar['aupr_out'], res_big_tar['f1_threshold'], res_big_tar['f1_accuracy'], res_big_tar['j_threshold'], res_big_tar['j_accuracy']
    print(f"SRC->TAR1:      AUROC: {auroc1:.4f}, FPR95: {fpr1:.4f}, AUPR_IN: {auprin1:.4f}, AUPR_OUT: {auprout1:.4f}, F1_TH: {f1_threshold1:.4f}, F1_TH_ACC: {f1_accuracy1:.4f}, J_TH: {j_threshold1:.4f}, J_TH_ACC: {j_accuracy1:.4f}")
    print(f"SRC->TAR2:      AUROC: {auroc2:.4f}, FPR95: {fpr2:.4f}, AUPR_IN: {auprin2:.4f}, AUPR_OUT: {auprout2:.4f}, F1_TH: {f1_threshold2:.4f}, F1_TH_ACC: {f1_accuracy2:.4f}, J_TH: {j_threshold2:.4f}, J_TH_ACC: {j_accuracy2:.4f}")
    print(f"SRC->TAR1+TAR2: AUROC: {auroc3:.4f}, FPR95: {fpr3:.4f}, AUPR_IN: {auprin3:.4f}, AUPR_OUT: {auprout3:.4f}, F1_TH: {f1_threshold3:.4f}, F1_TH_ACC: {f1_accuracy3:.4f}, J_TH: {j_threshold3:.4f}, J_TH_ACC: {j_accuracy3:.4f}")


def cos_sim(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def get_sncore_id_ood_loaders(opt):
    """

    Args:
        opt: script arguments

    Returns:
        train_loader, (src_loader, tar1_loader, tar2_loader), (src_name, tar1_name, tar2_name)

    """
    # no augmentation applied at test time
    apply_transf = None
    apply_shuffle = False
    apply_drop_last = False

    # train loader to compute category centroids
    train_dataset = ShapeNetCore4k(
        data_root=opt.data_root,
        split="train",
        class_choice=list(eval(opt.src).keys()),
        num_points=opt.num_points,
        transforms=apply_transf)

    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        worker_init_fn=init_np_seed,
        drop_last=apply_drop_last,
        shuffle=apply_shuffle)

    # test loaders (ID)
    src_dataset = ShapeNetCore4k(
        data_root=opt.data_root,
        split="test",
        class_choice=list(eval(opt.src).keys()),
        num_points=opt.num_points,
        transforms=apply_transf)

    src_loader = DataLoader(
        src_dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        worker_init_fn=init_np_seed,
        drop_last=apply_drop_last,
        shuffle=apply_shuffle)

    # target loaders (OOD)
    # target #1
    tar1_dataset = ShapeNetCore4k(
        data_root=opt.data_root,
        split="test",
        class_choice=list(eval(opt.tar1).keys()),
        num_points=opt.num_points,
        transforms=apply_transf)

    tar1_loader = DataLoader(
        tar1_dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        worker_init_fn=init_np_seed,
        drop_last=apply_drop_last,
        shuffle=apply_shuffle)

    # target #2
    tar2_dataset = ShapeNetCore4k(
        data_root=opt.data_root,
        split="test",
        class_choice=list(eval(opt.tar1).keys()),
        num_points=opt.num_points,
        transforms=apply_transf)

    tar2_loader = DataLoader(
        tar2_dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        worker_init_fn=init_np_seed,
        drop_last=apply_drop_last,
        shuffle=apply_shuffle)

    print(f"\nSrc: {opt.src}, Tar 1: {opt.tar1}, Tar 2: {opt.tar2}, Tar 3: {opt.tar1}+{opt.tar2}")
    return train_loader, (src_loader, tar1_loader, tar2_loader), (opt.src, opt.tar1, opt.tar2)


@torch.no_grad()
def get_simclr_proj(model, loader):
    """ DDP impl """
    all_proj = []
    all_labels = []
    model.eval()
    for i, batch in enumerate(loader, 0):
        points, labels = batch[0], batch[1]
        # not expecting list views but batch of points [bs,num_points,3]
        assert torch.is_tensor(points)
        points = points.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        z = model(points)
        if is_dist() and get_ws() > 1:
            z = gather(z, dim=0)
            labels = gather(labels, dim=0)
        all_labels.append(labels)
        all_proj.append(z)
    all_proj = torch.cat(all_proj)
    all_labels = torch.cat(all_labels)
    return all_proj, all_labels


@torch.no_grad()
def get_network_output(model, loader, softmax=True):
    """ DDP impl """
    all_logits = []
    all_pred = []
    all_labels = []
    model.eval()
    for i, batch in enumerate(tqdm(loader, disable=DISABLE_TQDM), 0):
        points, labels = batch[0], batch[1]
        assert torch.is_tensor(points), "expected BNC tensor as batch[0]"
        points = points.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        logits = model(points)
        if is_dist() and get_ws() > 1:
            logits = gather(logits, dim=0)
            labels = gather(labels, dim=0)

        all_logits.append(logits)
        probs = F.softmax(logits, 1) if softmax else logits
        _, pred = logits.data.max(1)
        all_pred.append(pred)
        all_labels.append(labels)
    all_logits = torch.cat(all_logits, dim=0)
    all_pred = torch.cat(all_pred, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return all_logits, all_pred, all_labels


@torch.no_grad()
def get_confidence(model, loader, softmax=True):
    """ DDP impl """
    all_conf = []
    all_pred = []
    all_labels = []
    model.eval()
    for i, batch in enumerate(tqdm(loader, disable=DISABLE_TQDM), 0):
        points, labels = batch[0], batch[1]
        assert torch.is_tensor(points), "expected BNC tensor as batch[0]"
        points = points.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        logits = model(points)
        if is_dist() and get_ws() > 1:
            logits = gather(logits, dim=0)
            labels = gather(labels, dim=0)
        probs = F.softmax(logits, 1) if softmax else logits
        conf, pred = probs.data.max(1)
        all_conf.append(conf)
        all_pred.append(pred)
        all_labels.append(labels)
    all_conf = torch.cat(all_conf, dim=0)
    all_pred = torch.cat(all_pred, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return all_conf, all_pred, all_labels


@torch.no_grad()
def get_penultimate_feats(model, loader):
    """ DDP impl """
    all_feats = []
    all_labels = []
    model.eval()
    for i, batch in enumerate(tqdm(loader, disable=DISABLE_TQDM), 0):
        points, labels = batch[0], batch[1]
        assert torch.is_tensor(points), "expected BNC tensor as batch[0]"
        points = points.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        feats = model(points, return_penultimate=True)
        if is_dist() and get_ws() > 1:
            feats = gather(feats, dim=0)
            labels = gather(labels, dim=0)
        all_feats.append(feats)
        all_labels.append(labels)
    all_feats = torch.cat(all_feats)
    all_labels = torch.cat(all_labels, dim=0)
    return all_feats, all_labels


def iterate_data_odin(model, loader, epsilon=0.0, temper=1000):
    """
    Source: https://github.com/deeplearning-wisc/gradnorm_ood
    """
    criterion = torch.nn.CrossEntropyLoss().cuda()
    confs = []
    for batch in tqdm(loader, disable=DISABLE_TQDM):
        x = batch[0]

        x = x.cuda()
        x.requires_grad = True
        outputs = model(x)

        maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)
        outputs = outputs / temper

        labels = torch.LongTensor(maxIndexTemp).cuda()
        loss = criterion(outputs, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(x.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        # Adding small perturbations to images
        tempInputs = torch.add(x.data, gradient, alpha=-epsilon)
        outputs = model(tempInputs)
        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

        confs.extend(np.max(nnOutputs, axis=1))

    return torch.tensor(np.array(confs))


def iterate_data_energy(model, loader, temper=1):
    """
    Source: https://github.com/deeplearning-wisc/gradnorm_ood
    """
    confs = []
    for batch in tqdm(loader, disable=DISABLE_TQDM):
        x = batch[0]

        with torch.no_grad():
            x = x.cuda()
            # compute output, measure accuracy and record loss.
            logits = model(x)

            conf = temper * torch.logsumexp(logits / temper, dim=1)
            confs.extend(conf.data.cpu().numpy())
    return torch.tensor(np.array(confs))


def iterate_data_gradnorm(model, loader, temperature=1):
    """
    Source: https://github.com/deeplearning-wisc/gradnorm_ood
    """
    confs = []
    logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()
    for batch in tqdm(loader, disable=DISABLE_TQDM):
        x = batch[0]
        inputs = x.cuda()

        model.zero_grad()
        outputs = model(inputs)
        targets = torch.ones((inputs.shape[0], outputs.shape[1])).cuda()
        outputs = outputs / temperature
        loss = torch.mean(torch.sum(-targets * logsoftmax(outputs), dim=-1))

        loss.backward()

        if isinstance(model.head, nn.Sequential):
            layer_grad = model.head[-1].weight.grad.data
        else:
            layer_grad = model.head.weight.grad.data

        layer_grad_norm = torch.sum(torch.abs(layer_grad)).cpu().numpy()
        confs.append(layer_grad_norm)

    return torch.tensor(np.array(confs))


@torch.no_grad()
def estimate_react_thres(model, loader, id_percentile=0.9):
    """
    Estimate the threshold to be used for react. 
    Strategy: choose threshold which allows to keep id_percentile% of 
    activations in in distribution data (source https://openreview.net/pdf?id=IBVBtz_sRSm).

    Args:
        model: base network 
        loader: in distribution data loader (some kind of validation set)
        id_percentile: percent of in distribution activations that we want to keep 

    Returns:
        threshold: float value indicating the computed threshold 
    """

    print("Estimating react threshold...", end="")
    id_activations = []

    for batch in tqdm(loader, disable=DISABLE_TQDM):
        x = batch[0]
        x = x.cuda()

        # we perform forward on modules separately so that we can access penultimate layer
        feats = model.enco(x)
        penultimate = model.penultimate(feats)
        id_activations.append(penultimate.cpu().view(-1).numpy())

    id_activations = np.concatenate(id_activations)
    thres = np.percentile(id_activations, id_percentile * 100)
    print(f"t = {thres:.4f}")

    return thres


def iterate_data_react(model, loader, threshold=1, energy_temper=1):
    """
    Source: https://github.com/deeplearning-wisc/gradnorm_ood
    """
    confs = []
    logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()
    for batch in tqdm(loader, disable=DISABLE_TQDM):
        x = batch[0]
        x = x.cuda()

        # we perform forward on modules separately so that we can access penultimate layer
        feats = model.enco(x)
        penultimate = model.penultimate(feats)
        # apply react
        x = penultimate.clip(max=threshold)
        logits = model.head(x)

        # apply energy 
        conf = energy_temper * torch.logsumexp(logits / energy_temper, dim=1)
        confs.extend(conf.data.cpu().numpy())

    return torch.tensor(np.array(confs))


def compute_centroids(model, train_loader):
    """ computes closed set centroids [num_classes x proj_dim] """
    num_classes = train_loader.dataset.num_classes
    # use centroids computed by averagin class clusters points
    train_proj, train_labels = get_simclr_proj(model, train_loader)
    centroids = []
    for cat in range(num_classes):
        cat_idxs = (train_labels == cat)
        proj_mean = train_proj[cat_idxs].mean(0).unsqueeze(0)
        centroids.append(proj_mean)
    centroids = torch.cat(centroids, 0).cuda()
    return centroids


def compute_clf_centroids(model, centroids, test_loader):
    """
    returns predictions and ground truth labels
    prediction is idx of closest centroid
    """
    feat, labels = get_simclr_proj(model, test_loader)
    feat = feat.to(centroids.device)
    sim = 0.5 + 0.5 * cos_sim(feat, centroids)
    _, preds = sim.max(-1)
    return preds, labels


def compute_sim_centroids(model, centroids, list_loaders):
    r"""
    For each loader computes the similarity of each of its samples to the closest category centroids
    Params:
        model: model to test
        centroids: tensor with precomputed closed set centroids
        list_loaders: list of data loaders for evaluation

    Returns:
        list of tensors, one for each loader in 'list_loaders'
    """

    list_scores = []  # for each loader in 'list_loaders' all similarity scores
    for loader in list_loaders:
        assert isinstance(loader, DataLoader), "expected list of dataloaders"
        feat, labels = get_simclr_proj(model, loader)
        del labels
        feat = feat.to(centroids.device)
        sim = 0.5 + 0.5 * cos_sim(feat, centroids)
        scores, _ = sim.max(-1)
        list_scores.append(scores)

    return list_scores


def print_fail_names(src_scores, tar_scores, src_names, tar_names, threshold):
    def fail_metrics(fail):
        values = {}
        for (lbl, pred, score) in fail:
            if (lbl, pred) not in values:
                values[(lbl, pred)] = (1, score, score, score)
            else:
                count, mean, min, max = values[(lbl, pred)]
                values[(lbl, pred)] = (count + 1, mean*count/(count+1) + score/(count+1), score if score < min else min, score if score > max else max)

        return dict(sorted(values.items(), key=lambda item: item[1], reverse=True))
    
    src_fail_names = [(src_names[i][0], src_names[i][1], src_scores[i]) for i in range(len(src_names)) if src_scores[i] < threshold]
    tar_fail_names = [(tar_names[i][0], tar_names[i][1], tar_scores[i]) for i in range(len(tar_names)) if tar_scores[i] >= threshold]

    src_fail_values = fail_metrics(src_fail_names)
    tar_fail_values = fail_metrics(tar_fail_names)

    print(f"Fail src: {len(src_fail_names)}/{len(src_scores)}")
    print(f"      Class\t| Pred\t| Times\t| Closest class\t| Mean\t| Min\t| Max")
    for (lbl, pred), (count, mean, min, max) in src_fail_values.items():
        print(f"      {lbl}\t| OOD\t| {count}\t| {pred}\t\t| {mean:.3f}\t| {min:.3f}\t| {max:.3f}")

    print(f"Fail tar: {len(tar_fail_names)}/{len(tar_scores)}")
    print(f"      Class\t| Pred\t| Times\t| Close class\t| Mean\t| Min\t| Max")
    for (lbl, pred), (count, mean, min, max) in tar_fail_values.items():
        print(f"      {lbl}\t| ID\t| {count}\t| {pred}\t\t| {mean:.3f}\t| {min:.3f}\t| {max:.3f}")

    return src_fail_values, tar_fail_values


def get_ood_metrics(src_scores, tar_scores, src_names, tar_names, src_label=1):
    """
    Computes ood metrics given src_scores and tar_scores
    Scores can be distances, confidences, ..
    """

    tar_label = int(not src_label)
    src_scores = to_numpy(src_scores)
    tar_scores = to_numpy(tar_scores)
    labels = np.concatenate([
        np.full(src_scores.shape[0], src_label, dtype=np.compat.long),
        np.full(tar_scores.shape[0], tar_label, dtype=np.compat.long)
    ], axis=0)
    scores = np.concatenate([src_scores, tar_scores], axis=0)
    res = calc_metrics(scores, labels)

    print(f"OOD F1 Test - Acc: {res['f1_accuracy']:.4f}, Th: {res['f1_threshold']:.4f}")
    print_fail_names(src_scores, tar_scores, src_names, tar_names, res['f1_threshold'])

    print(f"OOD J Test - Acc: {res['j_accuracy']:.4f}, Th: {res['j_threshold']:.4f}")
    print_fail_names(src_scores, tar_scores, src_names, tar_names, res['j_threshold'])

    return res


def eval_ood_sncore(scores_list, preds_list=None, labels_list=None, label_names_list=None, src_label=1, silent=False):
    """
    conf_list: [SRC, TAR1, TAR2]
    preds_list: [SRC, TAR1, TAR2]
    labels_list: [SRC, TAR1, TAR2]
    src_label: label for known samples when computing AUROC
    silent: if True does not print anything
    """

    if labels_list is None:
        labels_list = [None, None, None]

    if preds_list is None:
        preds_list = [None, None, None]
    
    if label_names_list is None:
        label_names_list = [None, None, None]

    tar_label = int(not src_label)

    if not silent:
        print(f"AUROC - Src label: {src_label}, Tar label: {tar_label}")

    src_conf, src_preds, src_labels, src_label_name = scores_list[0], preds_list[0], labels_list[0], label_names_list[0]
    tar1_conf, tar1_preds, tar1_labels, tar1_label_name = scores_list[1], preds_list[1], labels_list[1], label_names_list[1]
    tar2_conf, tar2_preds, tar2_labels, tar2_label_name = scores_list[2], preds_list[2], labels_list[2], label_names_list[2]
    

    # compute ID test accuracy
    src_acc, src_bal_acc = -1, -1
    if src_preds is not None:
        assert src_labels is not None
        src_labels = to_numpy(src_labels)
        src_preds = to_numpy(src_preds)
        if src_label_name is not None:    
            fail_names = [(src_label_name[lbl], src_label_name[pred]) for lbl, pred in zip(src_labels, src_preds) if lbl != pred]
            fail_counter = collections.Counter(fail_names)
        src_acc = skm.accuracy_score(src_labels, src_preds)
        src_bal_acc = skm.balanced_accuracy_score(src_labels, src_preds)
        if not silent:
            print(f"Src Test - Clf Acc: {src_acc:.4f}, Clf Bal Acc: {src_bal_acc:.4f}")
            if src_label_name is not None:
                print(f"Src Test - Fail: {len(fail_names)}/{len(src_labels)}")
                for (lbl, pred), value in fail_counter.most_common():
                    print(f"\tA {lbl} is predicted as {pred} - {value} times")

    src_names = [(src_label_name[lbl], src_label_name[pred]) for lbl, pred in zip(src_labels, src_preds)]
    tar1_names = [(tar1_label_name[lbl], src_label_name[pred]) for lbl, pred in zip(tar1_labels, tar1_preds)]
    tar2_names = [(tar2_label_name[lbl - 400], src_label_name[pred]) for lbl, pred in zip(tar2_labels, tar2_preds)]

    # Src vs Tar 1
    if not silent:
        print("Test Tar1")
    res_tar1 = get_ood_metrics(src_conf, tar1_conf, src_names, tar1_names, src_label)

    # Src vs Tar 2
    if not silent:
        print("Test Tar2")
    res_tar2 = get_ood_metrics(src_conf, tar2_conf, src_names, tar2_names, src_label)

    # Src vs Tar 1 + Tar 2
    if not silent:
        print("Test Tar1+Tar2")
    big_tar_conf = np.concatenate([to_numpy(tar1_conf), to_numpy(tar2_conf)], axis=0)
    big_tar_names = np.concatenate([tar1_names, tar2_names], axis=0)
    res_big_tar = get_ood_metrics(src_conf, big_tar_conf, src_names, big_tar_names, src_label)

    # N.B. get_ood_metrics reports inverted AUPR_IN and AUPR_OUT results
    # as we use label 1 for IN-DISTRIBUTION and thus we consider it positive. 
    # the ood_metrics library argue to use 

    if not silent:
        print_ood_output(res_tar1, res_tar2, res_big_tar)
        print(f"to spreadsheet: "
              f"{res_tar1['auroc']},{res_tar1['fpr_at_95_tpr']},{res_tar1['aupr_in']},{res_tar1['aupr_out']},"
              f"{res_tar2['auroc']},{res_tar2['fpr_at_95_tpr']},{res_tar2['aupr_in']},{res_tar2['aupr_out']},"
              f"{res_big_tar['auroc']},{res_big_tar['fpr_at_95_tpr']},{res_big_tar['aupr_in']},{res_big_tar['aupr_out']}")

    return src_acc, src_bal_acc, res_tar1, res_tar2, res_big_tar


def eval_ood_sncore_csi(model, train_loader, src_loader, tar1_loader, tar2_loader, use_norm=True):
    """ Using nearest training sample instead of category centroid to compute scores """

    eps = 1e-8
    train_feats, _ = get_simclr_proj(model, train_loader)

    src_feats, _ = get_simclr_proj(model, src_loader)
    src_norm = src_feats.norm(p=2.0, dim=1).clamp_min(eps)

    tar1_feats, _ = get_simclr_proj(model, tar1_loader)
    tar1_norm = tar1_feats.norm(p=2.0, dim=1).clamp_min(eps)

    tar2_feats, _ = get_simclr_proj(model, tar2_loader)
    tar2_norm = tar2_feats.norm(p=2.0, dim=1).clamp_min(eps)

    # cosine sim
    src_sim = cos_sim(src_feats, train_feats)
    src_scores, ids = src_sim.max(-1)
    if use_norm:
        src_scores = src_scores * src_norm

    tar1_sim = cos_sim(tar1_feats, train_feats)
    tar1_scores, ids = tar1_sim.max(-1)
    if use_norm:
        tar1_scores = tar1_scores * tar1_norm

    tar2_sim = cos_sim(tar2_feats, train_feats)
    tar2_scores, ids = tar2_sim.max(-1)
    if use_norm:
        tar2_scores = tar2_scores * tar2_norm

    res = eval_ood_sncore(
        scores_list=[src_scores.cpu(), tar1_scores.cpu(), tar2_scores.cpu()],
        preds_list=[None, None, None],
        labels_list=[None, None, None],
        src_label=1  # confidence should be higher for ID samples
    )

    return res
