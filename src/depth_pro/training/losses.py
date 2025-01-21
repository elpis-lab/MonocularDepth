import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_msssim import SSIM
import logging

logger = logging.getLogger(__name__)

class DepthLoss(nn.Module):
    def __init__(self, precision=torch.float32):
        super().__init__()
        self.precision = precision
        
    def compute_gradients(self, x, mask=None):
        x = x.to(self.precision)
        if len(x.shape) == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif len(x.shape) == 3:
            x = x.unsqueeze(1)
            
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device).to(self.precision).reshape(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=x.device).to(self.precision).reshape(1, 1, 3, 3)
        
        grad_x = F.conv2d(x, sobel_x, padding=1)
        grad_y = F.conv2d(x, sobel_y, padding=1)
        
        grads = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        if mask is not None:
            grads = grads * mask.to(self.precision).unsqueeze(1)
        return grads

    def mean_absolute_gradient_error(self, pred, target, mask=None, scales=6):
        loss = 0
        pred = pred.unsqueeze(1) if len(pred.shape) == 3 else pred
        target = target.unsqueeze(1) if len(target.shape) == 3 else target
        
        if mask is not None:
            mask = mask.to(self.precision).unsqueeze(1)
        
        for j in range(scales):
            if j > 0:
                pred = F.avg_pool2d(pred, 2)
                target = F.avg_pool2d(target, 2)
                if mask is not None:
                    mask = F.avg_pool2d(mask, 2)
                    
            grad_pred = self.compute_gradients(pred, mask)
            grad_target = self.compute_gradients(target, mask)
            
            if mask is not None:
                loss += (torch.abs(grad_pred - grad_target) * mask).sum() / (mask.sum() + 1e-6)
            else:
                loss += torch.abs(grad_pred - grad_target).mean()
            
        return loss / scales

    def compute_laplacian(self, x, mask=None):
        x = x.to(self.precision)
        if len(x.shape) == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif len(x.shape) == 3:
            x = x.unsqueeze(1)
            
        laplace_k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=x.device).to(self.precision).reshape(1, 1, 3, 3)
        lap = F.conv2d(x, laplace_k, padding=1)
        if mask is not None:
            lap = lap * mask.to(self.precision).unsqueeze(1)
        return lap

    def trimmed_mae_loss(self, pred, target, mask=None, trim_percentage=20):
        pred, target = pred.to(self.precision), target.to(self.precision)
        error = torch.abs(pred - target)
        if mask is not None:
            error = error * mask.to(self.precision)
            k = int(0.8 * mask.sum())
            return error[mask > 0].topk(k, largest=False)[0].mean()
        k = int(0.8 * error.numel())
        return error.flatten().topk(k, largest=False)[0].mean()
    
    def scale_invariant_loss(self, pred, target, mask=None):
        """Implements scale-invariant loss from the paper"""
        # Add small epsilon to prevent log(0)
        pred = torch.log(pred + 1e-6)
        target = torch.log(target + 1e-6)
        
        if mask is not None:
            n = mask.sum()
            diff = (pred - target) * mask
        else:
            n = pred.numel()
            diff = pred - target
            
        loss = (diff ** 2).sum() / n - (diff.sum() ** 2) / (n ** 2)
        return loss
    
    def create_mask(self, depth, null_value=0):
        min_depth, max_depth = 0.1, 10.0
        valid_mask = (depth > min_depth) & (depth < max_depth)
        return valid_mask.float()

    def __call__(self, pred, target, mask=None, stage="stage1"):
        pred = pred.to(self.precision)
        target = target.to(self.precision)
        
        if len(pred.shape) == 2:
            pred = pred.unsqueeze(0)
        if len(target.shape) == 2:
            target = target.unsqueeze(0)
        if len(pred.shape) == 4:
            pred = pred.squeeze(1)
        if len(target.shape) == 4:
            target = target.squeeze(1)
        if len(pred.shape) == 5:
            pred = pred.squeeze(1)
        if len(target.shape) == 5:
            target = target.squeeze(1)

        if mask is None:
            mask = self.create_mask(target, null_value=0)
            mask = mask.to(self.precision)

        mae_loss = self.trimmed_mae_loss(pred, target, mask)
        mage_loss = self.mean_absolute_gradient_error(pred, target, mask)

        if stage == "stage1":
            total_loss = mae_loss + 0.05 * self.scale_invariant_loss(pred, target, mask)

            return mae_loss, {
                "mae_loss": mae_loss.item(),
                "mage_loss": mage_loss.item(),
                "total_loss": total_loss.item()
            }

        male_loss = torch.abs(self.compute_laplacian(pred, mask) - self.compute_laplacian(target, mask)).mean()
        
        total_loss = mae_loss + mage_loss + 0.1 * male_loss
        
        return mae_loss, {
            "mae_loss": mae_loss.item(),
            "mage_loss": mage_loss.item(),
            "male_loss": male_loss.item(),
            "total_loss": total_loss.item()
        }