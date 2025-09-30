import numpy as np
import ot
import pandas as pd
import torch
from fairret.statistic import *
from sklearn.metrics import auc, roc_curve, accuracy_score

from humancompatible.train.benchmark.constraints.constraint_fns import *


def fair_stats(p_1, y_1, p_2, y_2):
    """
    Compute Independence, Separation, Inaccuracy, Sufficiency.
    """
    p = torch.concat([torch.tensor(p_1), torch.tensor(p_2)])
    w_onehot = torch.tensor([[0.0, 1.0]] * len(p_1))
    b_onehot = torch.tensor([[1.0, 0.0]] * len(p_2))
    sens = torch.vstack([w_onehot, b_onehot])
    labels = torch.concat([torch.tensor(y_1), torch.tensor(y_2)]).unsqueeze(1)
    pr0, pr1 = PositiveRate()(p, sens)
    fpr0, fpr1 = FalsePositiveRate()(p, sens, labels)
    tpr0, tpr1 = TruePositiveRate()(p, sens, labels)
    tnr0, tnr1 = 1 - fpr0, 1 - fpr1
    fnr0, fnr1 = 1 - tpr0, 1 - tpr1
    acc0, acc1 = Accuracy()(p, sens, labels)
    ppv0, ppv1 = PositivePredictiveValue()(p, sens, labels)
    fomr0, fomr1 = FalseOmissionRate()(p, sens, labels)
    # npv0, npv1 = 1 - fomr0, 1 - fomr1

    predictions = (p_1 >= 0.5).astype(float).flatten()
    tpr = (predictions @ y_1) / sum(y_1)
    tnr = ((-1*predictions + 1) @ (-1*y_1 + 1)) / sum(-1*y_1+1)
    fpr = 1-tnr
    fnr = 1 - tpr

    ind = abs(pr0 - pr1)
    sp = abs(tpr0 - tpr1) + abs(fpr0 - fpr1)

    ina = sum(np.concatenate([p_1, p_2]).flatten() != np.concatenate([y_1, y_2])) / (
        len(p_1) + len(p_2)
    )
    sf = abs(ppv0 - ppv1) + abs(fomr0 - fomr1)
    return ind, sp, ina, sf, tpr0, tpr1


@torch.inference_mode()
def make_groupwise_stats_table(X, y, loaded_models, full_preds=None):
    results_list = []
    criterion = torch.nn.BCEWithLogitsLoss()

    for model_index, model_iter in enumerate(loaded_models):
        (model_name, model) = model_iter

        alg = str.join("", model_name.split("_trial")[:-1])
        predictions = model(X)
        y = y.squeeze().to(float)
        predictions = predictions.squeeze()
        loss = criterion(predictions.squeeze(), y).cpu().numpy()
        predictions = torch.nn.functional.sigmoid(predictions)
        fpr, tpr, thresholds = roc_curve(
            y.cpu().numpy(), predictions.cpu().numpy()
        )
        auc_score = auc(fpr, tpr)
        acc = accuracy_score(y_pred = predictions > 0.5, y_true = y)
        tpr_fairret = TruePositiveRate()(predictions.unsqueeze(1), None, y.unsqueeze(1))
        pr_fairret = PositiveRate()(predictions.unsqueeze(1), None)
        predictions = (predictions >= 0.5).to(float)
        tpr = (predictions @ y) / sum(y)
        tnr = ((-1*predictions + 1) @ (-1*y + 1)) / sum(-1*y+1)
        fpr = 1-tnr
        fnr = 1 - tpr

        ppv = tpr / (tpr+fpr)
        fomr = fnr / (tnr + fnr)
        pr = sum(predictions)/len(predictions)

        results_list.append(
            {
                "Model": str(model_name),
                "Algorithm": alg,
                "acc": acc,
                "auc": auc_score,
                "fpr": fpr,
                "tpr_fairret": tpr_fairret,
                "tpr": tpr,
                "ppv": ppv,
                "fomr": fomr,
                "pr": pr,
                "pr_fairret": pr_fairret,
                "loss": loss
            }
        )
    return pd.DataFrame(results_list)
        
        # make table of "deviation from overall rate"


@torch.inference_mode()
def make_pairwise_constraint_stats_table(X_0, y_0, X_1, y_1, loaded_models):
    results_list = []
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for model_index, model_iter in enumerate(loaded_models):
        (model_name, model) = model_iter

        alg = str.join("", model_name.split("_trial")[:-1])
        predictions_0 = model(X_0)
        predictions_1 = model(X_1)
        if torch.any(torch.isnan(predictions_0)) or torch.any(
            torch.isnan(predictions_1)
        ):
            print(f"skipped {model_name}")
            continue
        y_0 = y_0.squeeze()
        y_1 = y_1.squeeze()
        l_0 = loss_fn(predictions_0.squeeze(), y_0).cpu().numpy()
        l_1 = loss_fn(predictions_1.squeeze(), y_1).cpu().numpy()
        predictions_0 = torch.nn.functional.sigmoid(predictions_0)
        predictions_1 = torch.nn.functional.sigmoid(predictions_1)
        # Calculate AUCs for sensitive attribute 0
        fpr_0, tpr_0, thresholds_0 = roc_curve(
            y_0.cpu().numpy(), predictions_0.cpu().numpy()
        )
        auc_0 = auc(fpr_0, tpr_0)
        # Calculate AUCs for sensitive attribute 1
        fpr_1, tpr_1, thresholds_1 = roc_curve(
            y_1.cpu().numpy(), predictions_1.cpu().numpy()
        )
        auc_1 = auc(fpr_1, tpr_1)
        auc_hm = (auc_0 * auc_1) / (auc_0 + auc_1)
        auc_m = (auc_0 + auc_1) / 2
        
        # Calculate TPR-FPR difference for sensitive attribute 0
        # tpr_minus_fpr_0 = tpr_0 - fpr_0
        # optimal_threshold_index_0 = np.argmax(tpr_minus_fpr_0)
        # optimal_threshold_0 = thresholds_0[optimal_threshold_index_0]

        # # Calculate TPR-FPR difference for sensitive attribute 1
        # tpr_minus_fpr_1 = tpr_1 - fpr_1
        # optimal_threshold_index_1 = np.argmax(tpr_minus_fpr_1)
        # optimal_threshold_1 = thresholds_1[optimal_threshold_index_1]

        p_0_np = (predictions_0 > 0.5).cpu().numpy()
        p_1_np = (predictions_1 > 0.5).cpu().numpy()
        y_w_np = y_0.cpu().numpy()
        y_nw_np = y_1.cpu().numpy()

        ind, sp, ina, sf, tpr0, tpr1 = fair_stats(p_0_np, y_w_np, p_1_np, y_nw_np)

        acc_0 = accuracy_score(
            y_true=y_0, y_pred=np.array([y > 0.5 for y in predictions_0])
        )
        acc_1 = accuracy_score(
            y_true=y_1, y_pred=np.array([y > 0.5 for y in predictions_1])
        )

        a0, x0 = np.histogram(predictions_0, bins=50)
        a1, x1 = np.histogram(predictions_1, bins=x0)
        a0 = a0.astype(float)
        a1 = a1.astype(float)
        a0 /= np.sum(a0)
        a1 /= np.sum(a1)
        wd = ot.wasserstein_1d(x0[1:], x1[1:], a0, a1, p=2)
        # Store results in the DataFrame
        results_list.append(
            {
                "Model": str(model_name),
                "Algorithm": alg,
                "AUC_M": auc_m,
                "Ind": ind,
                "Sp": sp,
                "Ina": ina,
                "Sf": sf,
                "Wd": wd,
                "|Loss_0 - Loss_1|": abs(l_0 - l_1),
                "|TPR_0 - TPR_1|": abs(tpr0 - tpr1),
                "acc_diff": abs(acc_0 - acc_1),
            }
        )

    res_df = pd.DataFrame(results_list)
    return res_df


def aggregate_model_stats_table(table: pd.DataFrame, agg_fns, agg_cols="Algorithm"):
    if len(agg_fns) == 1 and not isinstance(agg_fns, str):
        df = table.drop("Model", axis=1).groupby(agg_cols).agg(agg_fns[0]).sort_index()
    else:
        df = table.drop("Model", axis=1).groupby(agg_cols).agg(agg_fns)

    df["Algname"] = df.apply(lambda row: row.name, axis=1)
    df["Algname"] = pd.Categorical(
        df["Algname"],
        [
            "SGD",
            "SGD + Fairret",
            "Stochastic Ghost",
            "ALM",
            "SSL-ALM",
            "Switching Subgradient",
        ],
    )
    df = df.sort_values(by="Algname", axis=0)
    return df