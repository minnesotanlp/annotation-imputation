import numpy as np
import torch

from sklearn.metrics import mean_squared_error


def hit(gt_item, pred_items):
    if gt_item in pred_items:
        return 1
    return 0


def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index + 2))
    return 0


def metrics(model, test_loader, top_k):
    HR, NDCG = [], []
    for user, item, label in test_loader:
        user = user.cuda()
        item = item.cuda()

        predictions = model(user, item)
        _, indices = torch.topk(predictions, top_k)
        recommends = torch.take(item, indices).cpu().numpy().tolist()

        gt_item = item[0].item()
        HR.append(hit(gt_item, recommends))
        NDCG.append(ndcg(gt_item, recommends))

    return np.mean(HR), np.mean(NDCG)


def rmse(model, test_loader, max_label):
    # assumes the labels are consecutive
    all_preds = []
    all_labels = []

    for user, item, label in test_loader:
        user = user.cuda()
        item = item.cuda()

        with torch.no_grad():
            predictions = model(user, item)

        all_labels.append(label)
        all_preds.append(predictions)
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_preds = torch.cat(all_preds, dim=0).cpu().numpy()
    # Round positive values to nearest integer, and round negative values up to 0
    all_preds = np.where(all_preds >= 0, np.round(all_preds).astype(np.int64), 0)
    # Round values larger than max_label to max_label
    all_preds = np.clip(all_preds, 0, max_label)
    return mean_squared_error(all_labels, all_preds, squared=False)


def get_imputed_outputs(model, anno_matrix, loader, max_label):
    all_preds = []
    all_labels = []

    all_users = []
    all_items = []

    for user, item, label in loader:
        user = user.cuda()
        item = item.cuda()

        with torch.no_grad():
            predictions = model(user, item)

        all_labels.append(label)
        all_preds.append(predictions)
        all_users.append(user)
        all_items.append(item)

    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_preds = torch.cat(all_preds, dim=0).cpu().numpy()
    all_users = torch.cat(all_users, dim=0).cpu().numpy()
    all_items = torch.cat(all_items, dim=0).cpu().numpy()

    all_preds = np.where(all_preds >= 0, np.round(all_preds).astype(np.int64), 0)
    all_preds = np.clip(all_preds, 0, max_label)
    anno_matrix[all_items, all_users] = all_preds

    return anno_matrix
