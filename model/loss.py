import torch.nn.functional as F
import torch
import torch.nn as nn

def nll_loss(output, target):
    return F.nll_loss(output, target)

def mse_loss(output, target):
    return F.mse_loss(output, target)

class CELoss(nn.Module):
    def __init__(self, reduction='mean', weight=None):
        super(CELoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction=reduction,
                                             weight=weight)

    def forward(self, input, target):
        input = input["prediction_out"]
        loss = self.criterion(input, target)
        return {"target_loss": loss}

class SelectiveNetLoss(torch.nn.Module):
    def __init__(
            self, iteration, CE, selection_threshold, lm, alpha, coverage: float,
            dataset="cub", device='cpu', arch=None
    ):
        """
        Based on the implementation of SelectiveNet
        Args:
            loss_func: base loss function. the shape of loss_func(x, target) shoud be (B).
                       e.g.) torch.nn.CrossEntropyLoss(reduction=none) : classification
            coverage: target coverage.
            lm: Lagrange multiplier for coverage constraint. original experiment's value is 32.
        """
        super(SelectiveNetLoss, self).__init__()
        assert 0.0 < coverage <= 1.0

        self.CE = CE
        self.coverage = coverage
        self.iteration = iteration
        self.selection_threshold = selection_threshold
        self.dataset = dataset
        self.arch = arch
        self.device = device
        self.lm = lm
        self.alpha = alpha

    def forward(
            self, outputs, target, prev_selection_outs=None):

        prediction_out = outputs["prediction_out"]
        selection_out = outputs["selection_out"]
        out_aux = outputs["out_aux"]
        target = target

        if prev_selection_outs is None:
            prev_selection_outs = []

        if self.iteration == 1:
            weights = selection_out
        else:
            pi = 1
            for prev_selection_out in prev_selection_outs:
                pi *= (1 - prev_selection_out)
            weights = pi * selection_out

        # if self.dataset == "cub" or self.dataset == "CIFAR10":
        #     if self.iteration > 1 and epoch >= 85:
        #         condition = torch.full(prev_selection_outs[0].size(), True).to(device)
        #         for proba in prev_selection_outs:
        #             condition = condition & (proba < self.selection_threshold)
        #         emp_coverage = torch.sum(weights) / (torch.sum(condition) + 1e-12)
        #     else:
        #         emp_coverage = torch.mean(weights)
        # elif self.dataset == "mimic_cxr":
        #     emp_coverage = torch.mean(weights)
        # elif self.dataset == "HAM10k" or self.dataset == "SIIM-ISIC":
        #     if self.iteration > 1:
        #         condition = torch.full(prev_selection_outs[0].size(), True).to(device)
        #         for proba in prev_selection_outs:
        #             condition = condition & (proba < self.selection_threshold)
        #         emp_coverage = torch.sum(weights) / (torch.sum(condition) + 1e-12)
        #     else:
        #         emp_coverage = torch.mean(weights)
        emp_coverage = torch.mean(weights)

        CE_risk = torch.mean(self.CE(prediction_out, target) * weights.view(-1))
        emp_risk = (CE_risk) / (emp_coverage + 1e-12)

        # compute penalty (=psi)
        coverage = torch.tensor([self.coverage], dtype=torch.float32, requires_grad=True, device=self.device)
        penalty = (torch.max(
            coverage - emp_coverage,
            torch.tensor([0.0], dtype=torch.float32, requires_grad=True, device=self.device),
        ) ** 2)
        penalty *= self.lm

        selective_loss = emp_risk + penalty

        # auxillary loss
        aux_loss = torch.mean(self.CE(out_aux, target))
        total_loss = self.alpha * selective_loss + (1 - self.alpha) * aux_loss
        return {
            "selective_loss": selective_loss,
            "emp_coverage": emp_coverage,
            "CE_risk": CE_risk,
            "emp_risk": emp_risk,
            "cov_penalty": penalty,
            "aux_loss": aux_loss,
            "target_loss": total_loss
        }
        # return total_loss
