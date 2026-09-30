import torch
import torch.nn as nn
import torch.nn.functional as F

class MedicalFocalLoss(nn.Module):
    """
    Focal Loss for Medical Image Classification:
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Reduces the relative loss for well-classified examples (e.g. background/clear skin)
    and puts more focus on hard, misclassified samples (e.g. early Melanoma, subtle Psoriasis).
    """
    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = 'mean'):
        super(MedicalFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
