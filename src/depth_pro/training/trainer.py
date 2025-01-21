# depth_pro/training/trainer.py

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import logging
import os

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        device,
        scheduler=None,
        checkpoint_dir=None
    ):
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        # Wrap model with DataParallel and move to device
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs")
            # self.model = nn.DataParallel(model).to(device)
            self.model = model.to(device)
        else:
            self.model = model.to(device)
            
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.best_val_loss = float('inf')
        self.epoch = 0
        self.scaler = GradScaler()
        
    def train_one_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        loss_dict_sum = {}
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                images = batch["image"].to(self.device, dtype=torch.float16)
                depths = batch["depth"].to(self.device, dtype=torch.float16)
                inverse_depths = batch["inverse_depth"].to(self.device, dtype=torch.float16)
                focal_lengths = batch["focal_length"].to(self.device, dtype=torch.float16)
                
                if self.epoch == 0 and total_loss == 0:  # Only log once at start
                    logger.info(f'Batch shapes:')
                    logger.info(f'images: {images.shape}, {images.dtype}')
                    logger.info(f'depths: {depths.shape}, {depths.dtype}')
                
                # Forward pass
                pred_inverse_depth, pred_fov = self.model(images)
                pred_depth = 1.0 / torch.clamp(pred_inverse_depth, min=1e-6)
                
                # Remove channel dimension
                pred_depth = pred_depth.squeeze(1)
                pred_inverse_depth = pred_inverse_depth.squeeze(1)
                
                # Compute loss
                loss, loss_dict = self.criterion(
                    pred_depth,
                    depths,
                    pred_inverse_depth,
                    inverse_depths
                )
            
            # Backward pass with gradient scaling
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            
            # Update weights with gradient scaling
            # self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            # self.scaler.step(self.optimizer)
            # self.scaler.update()
            
            # Update metrics
            total_loss += loss.item()
            for k, v in loss_dict.items():
                loss_dict_sum[k] = loss_dict_sum.get(k, 0) + v
            
            # Update progress bar
            postfix = {
                "loss": f"{loss.item():.4f}",
                "depth_loss": f"{loss_dict.get('depth_loss', 0):.4f}",
                "grad_loss": f"{loss_dict.get('gradient_loss', 0):.4f}"
            }
            pbar.set_postfix(postfix)
        
        # Compute averages
        avg_loss = total_loss / len(train_loader)
        avg_loss_dict = {k: v / len(train_loader) for k, v in loss_dict_sum.items()}
        
        self.epoch += 1
        return avg_loss, avg_loss_dict
        
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        loss_dict_sum = {}
        
        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                images = batch["image"].to(self.device, dtype=torch.float16)
                depths = batch["depth"].to(self.device, dtype=torch.float16)
                inverse_depths = batch["inverse_depth"].to(self.device, dtype=torch.float16)
                focal_lengths = batch["focal_length"].to(self.device, dtype=torch.float16)
                
                # Forward pass
                pred_inverse_depth, pred_fov = self.model(images)
                pred_depth = 1.0 / torch.clamp(pred_inverse_depth, min=1e-6)
                
                # Remove channel dimension
                pred_depth = pred_depth.squeeze(1)
                pred_inverse_depth = pred_inverse_depth.squeeze(1)
                
                # Compute loss
                loss, loss_dict = self.criterion(
                    pred_depth,
                    depths,
                    pred_inverse_depth,
                    inverse_depths
                )
                
                total_loss += loss.item()
                for k, v in loss_dict.items():
                    loss_dict_sum[k] = loss_dict_sum.get(k, 0) + v
                    
        avg_loss = total_loss / len(val_loader)
        avg_loss_dict = {k: v / len(val_loader) for k, v in loss_dict_sum.items()}
        
        return avg_loss, avg_loss_dict
        
    def save_checkpoint(self, val_loss):
        """Save model checkpoint if validation loss improves."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            checkpoint = {
                'epoch': self.epoch,
                'model_state_dict': self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'val_loss': val_loss,
            }
            torch.save(
                checkpoint,
                f"{self.checkpoint_dir}/model_epoch_{self.epoch}_loss_{val_loss:.4f}.pt"
            )
            logger.info(f"Saved new best model with validation loss {val_loss:.4f}")
            
    def train(self, train_loader, val_loader, num_epochs):
        """Full training loop."""
        logger.info("Starting training")
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_loss, train_loss_dict = self.train_one_epoch(train_loader)
            
            # Validate
            val_loss, val_loss_dict = self.validate(val_loader)
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
                
            # Save checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(val_loss)
            # self.save_checkpoint(val_loss)