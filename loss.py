import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=0.95, gamma=1.5, reduction='mean'):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        
        p_t = p * targets + (1 - p) * (1 - targets)
        
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        focal_weight = (1 - p_t) ** self.gamma
        loss = focal_weight * bce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
            
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class HybridBCEMedicalLoss(nn.Module):
    def __init__(self, pos_weight_bce=20.0, alpha_focal=0.95, gamma_focal=1.5, lambda_hybrid=0.5):
        super(HybridBCEMedicalLoss, self).__init__()
        self.lambda_hybrid = lambda_hybrid

        pos_weight_tensor = torch.tensor([pos_weight_bce], dtype=torch.float32)
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        
        self.focal_loss = WeightedFocalLoss(alpha=alpha_focal, gamma=gamma_focal)

    def forward(self, inputs, targets):
        targets = targets.float().view_as(inputs)
        
        loss_bce = self.bce_loss(inputs, targets)
        loss_focal = self.focal_loss(inputs, targets)
        
        total_loss = (self.lambda_hybrid * loss_bce) + ((1 - self.lambda_hybrid) * loss_focal)
        return total_loss



criterion = HybridBCEMedicalLoss(
    pos_weight_bce=20.0,  # بر اساس نسبت داده‌های بیمارستان‌هایت
    alpha_focal=0.95,     # اهمیت دادن بیشتر به کلاس مثبت در فوکال لوس
    gamma_focal=1.0,      # گامای ملایم ۱ برای کنترل نویزهای پزشکی
    lambda_hybrid=0.5     # سهم ۵۰-۵۰ برای هر دو تابع زیان
)

