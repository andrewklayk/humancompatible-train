import torch
from fairret.statistic import (
    TruePositiveRate,
    FalseNegativeFalsePositiveFraction,
    FalsePositiveRate,
    PositiveRate,
    Accuracy,
)
from fairret.loss import NormLoss


def tpr_equality(_, net, c_data):
    statistic = TruePositiveRate()
    loss = NormLoss(statistic, p=1)

    return fairret_stat_equality(net, c_data, loss)


def ppv_equality(_, net, c_data):
    statistic = FalseNegativeFalsePositiveFraction()
    loss = NormLoss(statistic, p=1)

    return fairret_stat_equality(net, c_data, loss)


def acc_equality(_, net, c_data):
    statistic = Accuracy()
    loss = NormLoss(statistic, p=1)

    return fairret_stat_equality(net, c_data, loss)


def fairret_stat_equality(net, c_data, loss):
    g1_inputs, g1_labels = c_data[0]
    g2_inputs, g2_labels = c_data[1]

    g1_outs = net(g1_inputs).squeeze()
    g2_outs = net(g2_inputs).squeeze()
    # if not (1 in a_labels or 1 in g2_labels):
    #     return torch.tensor(0)

    group_codes = [0] * len(g1_labels) + [1] * len(g2_labels)
    group_codes = torch.tensor(
        [[0.0, 1.0] if x == 1 else [1.0, 0.0] for x in group_codes]
    )

    return loss(
        torch.concat([g1_outs, g2_outs]).unsqueeze(1),
        group_codes,
        torch.concat([g1_labels, g2_labels]).unsqueeze(1),
    )


def dummy(_, net, c_data):
    r = torch.zeros(1)
    r.grad = 0
    return r


def loss_equality(loss, net, c_data):
    g1_inputs, g1_labels = c_data[0]
    g2_inputs, g2_labels = c_data[1]
    g1_outs = net(g1_inputs)
    if g1_labels.ndim == 0:
        g1_labels = g1_labels.reshape(1)
        g2_labels = g2_labels.reshape(1)
    if g1_labels.ndim < g1_outs.ndim:
        g1_labels = g1_labels.unsqueeze(1)
        g2_labels = g2_labels.unsqueeze(1)
    g1_loss = loss(g1_outs, g1_labels)
    g2_outs = net(g2_inputs)
    g2_loss = loss(g2_outs, g2_labels)

    val = g1_loss - g2_loss
    return val



def abs_diff_tpr(_, net, c_data):#, stat):
    tpr = TruePositiveRate()
    g1_inputs, g1_labels = c_data[0]
    g2_inputs, g2_labels = c_data[1]
    g1_outs = torch.nn.functional.sigmoid(net(g1_inputs))
    g2_outs = torch.nn.functional.sigmoid(net(g2_inputs))
    
    g1_pos_pred_mask = (g1_outs >= 0).squeeze()
    g2_pos_pred_mask = (g2_outs >= 0).squeeze()

    if g1_labels.ndim == 0:
        g1_labels = g1_labels.reshape(1)
        g2_labels = g2_labels.reshape(1)
    if g1_labels.ndim < g1_outs.ndim:
        g1_labels = g1_labels.unsqueeze(1)
        g2_labels = g2_labels.unsqueeze(1)

    g1_loss = tpr(g1_outs[g1_pos_pred_mask], None, g1_labels[g1_pos_pred_mask])
    g2_loss = tpr(g2_outs[g2_pos_pred_mask], None, g2_labels[g2_pos_pred_mask])

    val = abs(g1_loss - g2_loss)

    return val

def abs_diff_pr(_, net, c_data):#, stat):
    pr = PositiveRate()
    g1_inputs, g1_labels = c_data[0]
    g2_inputs, g2_labels = c_data[1]
    g1_outs = torch.nn.functional.sigmoid(net(g1_inputs))
    g2_outs = torch.nn.functional.sigmoid(net(g2_inputs))

    if g1_labels.ndim == 0:
        g1_labels = g1_labels.reshape(1)
        g2_labels = g2_labels.reshape(1)
    if g1_labels.ndim < g1_outs.ndim:
        g1_labels = g1_labels.unsqueeze(1)
        g2_labels = g2_labels.unsqueeze(1)

    g1_loss = pr(g1_outs, None)
    g2_loss = pr(g2_outs, None)

    val = abs(g1_loss - g2_loss)

    return val

def abs_diff_fpr(_, net, c_data):#, stat):
    tpr = FalsePositiveRate()
    g1_inputs, g1_labels = c_data[0]
    g2_inputs, g2_labels = c_data[1]
    g1_outs = torch.nn.functional.sigmoid(net(g1_inputs))
    g2_outs = torch.nn.functional.sigmoid(net(g2_inputs))

    if g1_labels.ndim == 0:
        g1_labels = g1_labels.reshape(1)
        g2_labels = g2_labels.reshape(1)
    if g1_labels.ndim < g1_outs.ndim:
        g1_labels = g1_labels.unsqueeze(1)
        g2_labels = g2_labels.unsqueeze(1)

    g1_loss = tpr(g1_outs, None, g1_labels)
    g2_loss = tpr(g2_outs, None, g2_labels)

    val = abs(g1_loss - g2_loss)

    return val

def abs_max_dev_from_overall_tpr(_, net, c_data):
    stats = []
    st = TruePositiveRate()
    for input, label in c_data:
        out = net(input)
        pred_sigm = torch.nn.functional.sigmoid(out)
        pos_pred_mask = (pred_sigm >= 0).squeeze()
        tpr = st(pred_sigm[pos_pred_mask], None, label[pos_pred_mask].unsqueeze(1))
        stats.append(tpr)

    stats = torch.cat(stats)
    all_inp = torch.cat([x[0] for x in c_data])
    all_lab = torch.cat([x[1] for x in c_data])
    all_out = net(all_inp)
    all_pred_sigm = torch.nn.functional.sigmoid(all_out)
    pos_pred_mask = (all_pred_sigm >= 0).squeeze()
    all_tpr = st(all_pred_sigm[pos_pred_mask], None, all_lab[pos_pred_mask].unsqueeze(1))

    val = torch.max(
        torch.abs(stats - all_tpr)
    )

    # val = torch.max(
    #     torch.abs(stats/all_tpr - 1)
    # )

    return val



def abs_max_dev_from_overall_fpr(_, net, c_data):
    stats = []
    st = FalsePositiveRate()
    for input, label in c_data:
        out = net(input)
        pred_sigm = torch.nn.functional.sigmoid(out)
        tpr = st(pred_sigm, None, label.unsqueeze(1))
        stats.append(tpr)

    stats = torch.cat(stats)
    all_inp = torch.cat([x[0] for x in c_data])
    all_lab = torch.cat([x[1] for x in c_data])
    all_out = net(all_inp)
    all_pred_sigm = torch.nn.functional.sigmoid(all_out)
    all_tpr = st(all_pred_sigm, None, all_lab.unsqueeze(1))

    val = torch.max(
        torch.abs(stats - all_tpr)
    )

    # val = torch.max(
    #     torch.abs(stats/all_tpr - 1)
    # )

    return val


def abs_loss_equality(loss, net, c_data):
    g1_inputs, g1_labels = c_data[0]
    g2_inputs, g2_labels = c_data[1]
    g1_outs = net(g1_inputs)
    if g1_labels.ndim == 0:
        g1_labels = g1_labels.reshape(1)
        g2_labels = g2_labels.reshape(1)
    if g1_labels.ndim < g1_outs.ndim:
        g1_labels = g1_labels.unsqueeze(1)
        g2_labels = g2_labels.unsqueeze(1)
    g1_loss = loss(g1_outs, g1_labels)
    g2_outs = net(g2_inputs)
    g2_loss = loss(g2_outs, g2_labels)

    val = g1_loss - g2_loss
    return torch.abs(val)


def fairret_constr(loss, net, c_data):
    g1_inputs, g1_labels = c_data[0]
    g2_inputs, g2_labels = c_data[1]
    g1_logits = net(g1_inputs)
    g2_logits = net(g2_inputs)
    g1_onehot = torch.tensor([[0.0, 1.0]] * len(g1_inputs))
    g2_onehot = torch.tensor([[1.0, 0.0]] * len(g2_inputs))
    logits = torch.concat([g1_logits, g2_logits])
    sens = torch.vstack([g1_onehot, g2_onehot])
    labels = torch.hstack([g1_labels, g2_labels]).unsqueeze(1)

    return loss(logits, sens, label=labels)


def fairret_pr_constr(loss, net, c_data):
    g1_inputs, _ = c_data[0]
    g2_inputs, _ = c_data[1]
    g1_logits = net(g1_inputs)
    g2_logits = net(g2_inputs)
    g1_onehot = torch.tensor([[0.0, 1.0]] * len(g1_inputs))
    g2_onehot = torch.tensor([[1.0, 0.0]] * len(g2_inputs))
    logits = torch.concat([g1_logits, g2_logits])
    sens = torch.vstack([g1_onehot, g2_onehot])

    return loss(logits, sens)