import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import logging
import os

logger = logging.getLogger(__name__)

def setup_ddp(rank, world_size):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

class DistributedTrainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        rank,
        world_size,
        accumulation_steps=4,  # Number of gradient accumulation steps
        scheduler=None,
        checkpoint_dir=None
    ):
        self.rank = rank
        self.world_size = world_size
        self.accumulation_steps = accumulation_steps
        
        # Move model to current GPU
        self.device = torch.device(f'cuda:{rank}')
        self.model = model.to(self.device)
        
        # Wrap model with DDP
        self.model = DDP(self.model, device_ids=[rank])
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        
        self.best_val_loss = float('inf')
        self.epoch = 0
        self.scaler = GradScaler()
        
        logger.info(f'Initialized trainer on GPU {rank} with {accumulation_steps} accumulation steps')

    def train_one_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        loss_dict_sum = {}
        
        # Reset gradients at the start of each epoch
        self.optimizer.zero_grad(set_to_none=True)
        
        if self.rank == 0:  # Only show progress bar on main process
            pbar = tqdm(train_loader, desc=f"Epoch {self.epoch}")
        else:
            pbar = train_loader
            
        for batch_idx, batch in enumerate(pbar):
            # Move data to current GPU
            with torch.cuda.amp.autocast():
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
                
                # Scale loss by accumulation steps
                loss = loss / self.accumulation_steps
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            
            # Update weights if we've accumulated enough gradients
            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            
            # Update metrics
            total_loss += loss.item() * self.accumulation_steps
            
            # Update loss dictionary
            for k, v in loss_dict.items():
                loss_dict_sum[k] = loss_dict_sum.get(k, 0) + v
                
            # Update progress bar on main process
            if self.rank == 0:
                postfix = {
                    "loss": f"{loss.item() * self.accumulation_steps:.4f}",
                    "depth_loss": f"{loss_dict.get('depth_loss', 0):.4f}",
                    "grad_loss": f"{loss_dict.get('gradient_loss', 0):.4f}"
                }
                pbar.set_postfix(postfix)
        
        # Handle any remaining gradients
        if (batch_idx + 1) % self.accumulation_steps != 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
        
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
                images = batch["image"].to(self.device, dtype=torch.float16)
                depths = batch["depth"].to(self.device, dtype=torch.float16)
                inverse_depths = batch["inverse_depth"].to(self.device, dtype=torch.float16)
                focal_lengths = batch["focal_length"].to(self.device, dtype=torch.float16)
                
                pred_inverse_depth, pred_fov = self.model(images)
                pred_depth = 1.0 / torch.clamp(pred_inverse_depth, min=1e-6)
                
                loss, loss_dict = self.criterion(
                    pred_depth,
                    depths,
                    pred_inverse_depth,
                    inverse_depths
                )
                
                total_loss += loss.item()
                for k, v in loss_dict.items():
                    loss_dict_sum[k] = loss_dict_sum.get(k, 0) + v
        
        # Synchronize validation loss across GPUs
        dist.all_reduce(torch.tensor(total_loss).to(self.device))
        for k in loss_dict_sum:
            dist.all_reduce(torch.tensor(loss_dict_sum[k]).to(self.device))
            
        avg_loss = total_loss / (len(val_loader) * self.world_size)
        avg_loss_dict = {k: v / (len(val_loader) * self.world_size) for k, v in loss_dict_sum.items()}
        
        return avg_loss, avg_loss_dict
        
    def save_checkpoint(self, val_loss):
        """Save model checkpoint if validation loss improves."""
        if self.rank == 0:  # Only save on main process
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                checkpoint = {
                    'epoch': self.epoch,
                    'model_state_dict': self.model.module.state_dict(),  # Save unwrapped model
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
        logger.info(f"Starting training on GPU {self.rank}")
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Set epoch for train sampler
            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)
            
            # Train
            train_loss, train_loss_dict = self.train_one_epoch(train_loader)
            
            # Validate
            val_loss, val_loss_dict = self.validate(val_loader)
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            
            # Save checkpoint
            self.save_checkpoint(val_loss)
            
            # Log metrics on main process
            if self.rank == 0:
                logger.info(f"Epoch {epoch}")
                logger.info(f"Train Loss: {train_loss:.4f}")
                for k, v in train_loss_dict.items():
                    logger.info(f"Train {k}: {v:.4f}")
                logger.info(f"Val Loss: {val_loss:.4f}")
                for k, v in val_loss_dict.items():
                    logger.info(f"Val {k}: {v:.4f}")
                logger.info("-" * 80)