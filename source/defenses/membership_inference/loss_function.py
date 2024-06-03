import torch.nn as nn
import torch
import torch.nn.functional as F


def get_loss(loss_type, device, args, num_classes = 10, reduction = "mean"):
    CONFIG = {
        "ce": nn.CrossEntropyLoss(),
        "focal": FocalLoss(gamma = args.alpha, beta = args.beta),
        "ccel":CCEL(alpha = args.alpha, beta = args.beta),
        "ccql":CCQL(alpha = args.alpha, beta = args.beta),
        "focal_ccel": FocalCCEL(alpha = args.alpha, beta = args.beta, gamma = args.gamma),
        }
    return CONFIG[loss_type]


def focal_loss(input_values, gamma, reduction="mean"):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values

    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError("Invalid reduction option. Use 'none', 'mean', or 'sum'.")
class FocalLoss(nn.Module):
    def __init__(self, gamma=0., beta = 1,reduction='mean'):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.beta = beta
        self.reduction = reduction
    def forward(self, input, target):
        return self.beta *focal_loss(F.cross_entropy(input, target, reduction="none"), self.gamma, reduction = self.reduction)
    
def taylor_exp(input_values, alpha, reduction="mean"):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = alpha*input_values - (1-alpha)*(p+torch.pow(p,2)/2)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError("Invalid reduction option. Use 'none', 'mean', or 'sum'.")

def concave_exp_loss(input_values, gamma =1, reduction="mean"):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = torch.exp(gamma*p)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError("Invalid reduction option. Use 'none', 'mean', or 'sum'.")
    
def ce_concave_exp_loss(input_values, alpha, beta):
    p = torch.exp(-input_values)
    
    loss = alpha * input_values - beta *torch.exp(p)
    return loss




class CCEL(nn.Module):
    def __init__(self, alpha = 0.5, beta = 1, gamma=1.0, tau =1, reduction='mean'):
        super(CCEL, self).__init__()
        assert gamma >= 1e-7
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.reduction_ = reduction
    def forward(self, input, target):
        # Calculate the cross-entropy loss without reduction
        ce_loss = F.cross_entropy(input, target, reduction="none")
        # Pass the calculated cross-entropy loss along with other parameters to your custom loss function
        # Ensure that the 'reduction' argument is not passed again if it's already expected by ce_concave_exp_loss function
        modified_loss = ce_concave_exp_loss(ce_loss, self.alpha, (1-self.alpha))
        # Apply the beta scaling and reduce the loss as needed
        if self.reduction_ == 'mean':
            return self.beta * modified_loss.mean()
        elif self.reduction_ == 'sum':
            return self.beta * modified_loss.sum()
        else:
            # If reduction is 'none', just return the scaled loss
            return self.beta * modified_loss


class CCQL(nn.Module):
    def __init__(self, alpha = 1,beta =1,reduction='mean'):
        super(CCQL, self).__init__()
        #assert gamma >= 1e-7
        self.gamma = 2
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
    def forward(self, input, target):
        ce = F.cross_entropy(input, target, reduction="none")
        return self.beta*taylor_exp(ce, self.alpha, self.reduction)



class FocalCCEL(nn.Module):
    def __init__(self, alpha = 1, beta = 1, gamma=1.0,reduction='mean'):
        super(FocalCCEL, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
    def forward(self, input, target):
        ce = F.cross_entropy(input, target, reduction="none")
        losses = focal_loss(ce, gamma = self.gamma, reduction="none")
        cel = concave_exp_loss(ce,reduction="none")
        loss = self.alpha * losses + (1-self.alpha)*cel
        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            return self.beta*loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError("Invalid reduction option. Use 'none', 'mean', or 'sum'.")