import os
import sys
import logging
import tempfile
from datetime import timedelta
from PIL import Image
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
from sklearn.metrics import root_mean_squared_error
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms
from src.depth_pro.network.encoder import DepthProEncoder
from src.depth_pro.network.decoder import MultiresConvDecoder
from src.depth_pro.training.dataset import DepthDataset, DepthAugmenter
from src.depth_pro.training.losses import DepthLoss
from src.depth_pro import create_model_and_transforms
from tqdm import tqdm
from typing import Mapping, Optional, Tuple, Union
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from torch.cuda import warnings as cuda_warnings
cuda_warnings.filterwarnings('ignore')

def setup(rank, world_size):
    """Sets up the distributed process group."""
    try:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=timedelta(seconds=120)
        )
        torch.cuda.set_device(rank)
        logger.info(f"Process {rank} initialized successfully")
    except Exception as e:
        logger.error(f"Process {rank} failed to initialize: {e}")
        raise

def cleanup():
    """Cleans up the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()

class DepthTrainer:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank}")
        self.scaler = torch.cuda.amp.GradScaler()
        
        self.min_valid_depth = 0.05
        self.max_valid_depth = 10.0

        self.precision = torch.float32

        # Stage-specific configurations
        self.stages = {
            'stage1': {
                'epochs': 50,
                # 'data_dir': "/home/badari/Thesis_Depth/Analysis/grasp1b_dataset_organized/",
                'data_dir': "/home/badari/Thesis_Depth/Analysis/Lab_Dataset_3k/",
                'encoder_lr': 1.28e-7, # NOTE : reduced 10x smaller
                'decoder_lr': 1.28e-6, # NOTE : reduced 10x smaller
                'batch_size': 1,
                'accumulation_steps': 128, # NOTE : changed from 128 to 32
            },
            'stage2': {
                'epochs': 0,
                'data_dir': "/home/badari/Thesis_Depth/Analysis/YCB_Seg/",
                'encoder_lr': 6.4e-6,
                'decoder_lr': 6.4e-5,
                'batch_size': 1,
                'accumulation_steps': 64,
            }
        }

        self.img_size = 1536

        self.config = {
            'resolution': (1536, 1536),
            'optimizer': 'Adam',
            'weight_decay': 0,
            'gradient_clip_norm': 0.1,
            'world_size': self.world_size,
        }

        self.current_stage = 'stage1'
        self.cur_epoch = 0
        
        # Initialize model and optimizer
        self.model_setup()
        self.model = DDP(
            self.model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=True,
            broadcast_buffers=False
        )

        # Initialize loss and metrics
        self.criterion = DepthLoss(precision=self.precision)
        self.best_val_loss = float('inf')
        self.patience = 10
        self.min_delta = 5e-5
        self.patience_counter = 0

        # Initialize wandb
        if self.rank == 0:
            wandb.init(project="depth-pro")
            # add config and stages to wandb
            wandb.config.update({'config': self.config , 'stages': self.stages})

        self.F_PX = {"640":607.7325439453125 , '1280':674.4 , '1536':1458.558 , '320':303.866 , '960':589.4914855957031} # 343.1580

        # # Load checkpoint first (includes best_val_loss)
        # self.load_checkpoint('checkpoints/stage1_depth_pro_12.pt')

    def load_checkpoint(self, checkpoint_path):
        """Load model weights and best_val_loss from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Keep the best validation loss from checkpoint
        self.best_val_loss = checkpoint['best_val_loss']
        # self.cur_epoch = checkpoint['epoch']
        self.current_stage = checkpoint['current_stage']
        
        if self.rank == 0:
            logger.info(f"Model weights loaded from checkpoint: {checkpoint_path}")
            logger.info(f"Keeping best validation loss: {self.best_val_loss}")
            logger.info(f"Starting training from epoch {self.cur_epoch} with new learning rates")
            logger.info(f"New encoder lr: {self.stages['stage1']['encoder_lr']}")
            logger.info(f"New decoder lr: {self.stages['stage1']['decoder_lr']}")

    def setup_training_stage(self, stage):
        """Configure training parameters for current stage."""
        self.current_stage = stage
        logger.info(f"\nStarting {stage}")

        # Update data loaders
        # self.data_setup(self.stages[stage]['data_dir'], is_synthetic=(stage=='stage2'))

        # Update learning rates
        for param_group in self.optimizer.param_groups:
            if 'encoder' in str(param_group['params'][0]):
                param_group['lr'] = self.stages[stage]['encoder_lr']
            else:
                param_group['lr'] = self.stages[stage]['decoder_lr']

        # Update training parameters
        self.config.update({
            'batch_size': self.stages[stage]['batch_size'],
            'gradient_clip_norm': 0.2 if stage == "stage1" else 0.1,
            'accumulation_steps': self.stages[stage]['accumulation_steps']
        })

        # Reset metrics
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        if self.rank == 0:
            logger.info(f"Learning rates - Encoder: {self.stages[stage]['encoder_lr']}, " 
                       f"Decoder: {self.stages[stage]['decoder_lr']}")
            
    def model_setup(self, model_path="../../checkpoints/depth_pro.pt"):
        """Initialize model, transforms, and optimizer."""
        self.model, self.transform = create_model_and_transforms(device=self.device, precision=self.precision)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device),
                strict=False
            )
        
        self.model = self.model.to(self.device)

        # self.model = torch.compile(self.model)

        # Make parameters contiguous
        for param in self.model.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()

        # Freeze LayerNorm layers
        self.freeze_layers()

        self.setup_optimizer_and_scheduler()

    def freeze_layers(self):
        """Freeze LayerNorm layers and adjust for current stage."""
        for name, param in self.model.named_parameters():
            if 'LayerNorm' in name:
                param.requires_grad = False
                if self.rank == 0:
                    logger.info(f"Freezing layer: {name}")

    def setup_optimizer_and_scheduler(self):
        """Initialize optimizer and scheduler with stage-specific settings."""
        self.optimizer = torch.optim.Adam([
            {'params': self.model.encoder.parameters(), 
             'lr': self.stages[self.current_stage]['encoder_lr'], 
             'weight_decay': self.config['weight_decay']},
            {'params': self.model.decoder.parameters(), 
             'lr': self.stages[self.current_stage]['decoder_lr'], 
             'weight_decay': self.config['weight_decay'] * 2},
            {'params': self.model.head.parameters(), 
             'lr': self.stages[self.current_stage]['decoder_lr'], 
             'weight_decay': self.config['weight_decay']}
        ])
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-8,
            verbose=True
        )

    def data_setup(self, data_dir, is_synthetic=False):
        """Initialize data loaders for current training stage."""
        # Create datasets with stage-specific settings
        logger.info(f"\nInitializing {'Synthetic' if is_synthetic else 'Real'} dataloaders , Data Dir : {data_dir}")
        train_dataset = DepthDataset(
            os.path.join(data_dir, "train"), 
            self.config['resolution'][0], 
            is_train=True,
            is_synthetic=is_synthetic
        )
        
        val_dataset = DepthDataset(
            os.path.join(data_dir, "val"), 
            self.config['resolution'][0], 
            is_train=False,
            is_synthetic=is_synthetic
        )

        # Setup distributed samplers
        self.train_sampler = DistributedSampler(
            train_dataset, 
            num_replicas=self.world_size, 
            rank=self.rank, 
            shuffle=True
        )
        
        self.val_sampler = DistributedSampler(
            val_dataset, 
            num_replicas=self.world_size, 
            rank=self.rank, 
            shuffle=True
        )

        # Create data loaders with stage-specific batch sizes
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.stages[self.current_stage]['batch_size'],
            sampler=self.train_sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.stages[self.current_stage]['batch_size'],
            sampler=self.val_sampler,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )

        # Calculate steps for learning rate scheduling
        total_steps = len(self.train_loader) * self.stages[self.current_stage]['epochs']
        self.warmup_steps = int(0.05 * total_steps)  # 5% warmup
        self.constant_steps = int(0.75 * total_steps)  # 75% constant
        self.decay_steps = int(0.20 * total_steps)  # 20% decay

        if self.rank == 0:
            logger.info(f"\nInitialized {'Synthetic' if is_synthetic else 'Real'} dataloaders:")
            logger.info(f"Train samples: {len(train_dataset)}")
            logger.info(f"Val samples: {len(val_dataset)}")
            logger.info(f"Batch size: {self.stages[self.current_stage]['batch_size']}")
            logger.info(f"Total steps per epoch: {len(self.train_loader)}")
            logger.info(f"Warmup steps: {self.warmup_steps}")
            logger.info(f"Constant steps: {self.constant_steps}")
            logger.info(f"Decay steps: {self.decay_steps}")

    def get_lr_scale(self, step):
        """Calculate learning rate scale based on current training step."""
        if step < self.warmup_steps:
            # Linear warmup
            return step / self.warmup_steps
        elif step < self.warmup_steps + self.constant_steps:
            # Constant learning rate
            return 1.0
        else:
            # Cosine decay
            decay_steps_elapsed = step - (self.warmup_steps + self.constant_steps)
            decay_total_steps = self.decay_steps
            cosine_decay = 0.5 * (1 + np.cos(np.pi * decay_steps_elapsed / decay_total_steps))
            return max(0.0, cosine_decay)
        
    def update_learning_rate(self, step):
        total_steps = len(self.train_loader) * self.stages[self.current_stage]['epochs']
        warmup_steps = int(0.01 * total_steps)  # 1% warmup
        
        if step < warmup_steps:
            lr_scale = step / warmup_steps
        else:
            # Cosine decay after warmup
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            lr_scale = 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['initial_lr'] * lr_scale

    # def update_learning_rate(self, step, stage='stage1'):
    #     total_steps = len(self.train_loader) * self.stages[stage]['epochs']
    #     warmup_steps = int(0.1 * total_steps)
    #     decay_steps = total_steps - warmup_steps

    #     if step < warmup_steps:
    #         lr_scale = step / warmup_steps
    #     else:
    #         lr_scale = 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / decay_steps))

    #     for param_group in self.optimizer.param_groups:
    #         base_lr = self.stages[stage]['encoder_lr'] if 'encoder' in param_group['params'][0].name else self.stages[stage]['decoder_lr']
    #         param_group['lr'] = base_lr * lr_scale

    def calculate_f_px(self, new_W, default_f_px, original_W=640):
        """
        Calculate the new focal length in pixels (f_px) for a resized image width.
        
        Parameters:
            new_W (float): The new image width.
            default_f_px (float): The default f_px value for the original width (default: 298.40 for W=640).
            original_W (float): The original image width (default: 640).
            
        Returns:
            float: The new f_px value for the resized image.
        """
        return default_f_px * (new_W / original_W)

    def calculate_canonical_inverse_depth(self, depth, f_px, W):
        """Improve depth normalization with null value handling"""
        # Create mask for valid depth values 
        valid_mask = (depth > self.min_valid_depth) & (depth < self.max_valid_depth)
        
        # Clamp depth values within valid range
        depth_map = torch.clamp(depth, self.min_valid_depth, self.max_valid_depth)
        
        # Compute inverse depth
        inverse_depth = 1.0 / depth_map
        
        canonical_inverse_depth = inverse_depth * (f_px / W)
        return canonical_inverse_depth, valid_mask

    def infer(
        self,
        x: torch.Tensor,
        f_px: Optional[Union[float, torch.Tensor]] = None,
        interpolation_mode="bilinear"
    ) -> Mapping[str, torch.Tensor]:
        """Infer depth and fov for a given image.

        If the image is not at network resolution, it is resized to 1536x1536 and
        the estimated depth is resized to the original image resolution.
        Note: if the focal length is given, the estimated value is ignored and the provided
        focal length is use to generate the metric depth values.

        Args:
        ----
            x (torch.Tensor): Input image
            f_px (torch.Tensor): Optional focal length in pixels corresponding to `x`.
            interpolation_mode (str): Interpolation function for downsampling/upsampling. 

        Returns:
        -------
            Tensor dictionary (torch.Tensor): depth [m], focallength [pixels].

        """
        # print(f'Infer Image shape : {x.shape} , len(x.shape) : {len(x.shape)}')
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        _, _, H, W = x.shape
        resize = H != self.img_size or W != self.img_size

        if resize:
            x = nn.functional.interpolate(
                x,
                size=(self.config['resolution']),
                mode=interpolation_mode,
                align_corners=False,
            )

        canonical_inverse_depth, fov_deg = self.model(x)
        
        if f_px is None:
            f_px = 0.5 * W / torch.tan(0.5 * torch.deg2rad(fov_deg.to(torch.float)))
        
        inverse_depth = canonical_inverse_depth * (W / f_px)
        f_px = f_px.squeeze()

        if resize:
            inverse_depth = nn.functional.interpolate(
                inverse_depth, size=(H, W), mode=interpolation_mode, align_corners=False
            )
            canonical_inverse_depth = nn.functional.interpolate(
                canonical_inverse_depth, size=(H, W), mode=interpolation_mode, align_corners=False
            )

        depth = 1.0 / torch.clamp(inverse_depth, min=self.min_valid_depth, max=self.max_valid_depth)

        return {
            "depth": depth.squeeze(),
            "canoical_inverse_depth": canonical_inverse_depth.squeeze(),
            "focallength_px": f_px,
        }

    def train_epoch(self, epoch, mode='train'):
        """Train/validate for one epoch."""
        total_loss = 0.0
        num_batches = 0
        
        loader = self.train_loader if mode == 'train' else self.val_loader
        self.model.train() if mode == 'train' else self.model.eval()
        
        with tqdm(total=len(loader), desc=f"{mode.capitalize()} Epoch {epoch}") as pbar:
            for batch_idx, batch in enumerate(loader):
                # Move data to device
                image = batch['image'].to(self.device)
                # convert to self.precision
                image = image.to(self.precision)
                depth = batch['depth'].to(self.device)
                depth = depth.to(self.precision)
                W = batch['W'].to(self.device)
                W = W.to(self.precision)

                f_px = torch.tensor(607.7325439453125).to(self.device)

                if mode == 'train':
                    step = len(loader) * epoch + batch_idx
                    # self.update_learning_rate(step)
                    with torch.amp.autocast('cuda') : preds = self.infer(image, f_px=f_px)
                else:
                    with torch.no_grad(): preds = self.infer(image, f_px=f_px)

                # f_px = preds['focallength_px']

                # canonical_inverse_depth = self.calculate_canonical_inverse_depth(depth, f_px, W)
                canonical_inverse_depth, valid_mask = self.calculate_canonical_inverse_depth(depth, f_px, W)
                
                # loss = self.criterion( preds['canoical_inverse_depth'] ,canonical_inverse_depth, stage=self.current_stage)
                loss = self.criterion( preds['canoical_inverse_depth'] ,canonical_inverse_depth, valid_mask, stage=self.current_stage)

                if mode == 'train':
                    # Backward pass and optimization
                    # scaled_loss = self.scaler.scale(loss)
                    scaled_loss = self.scaler.scale(loss[0])
                    scaled_loss.backward()
                    
                    if (batch_idx + 1) % self.config['accumulation_steps'] == 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config['gradient_clip_norm']
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)

                # Update metrics
                total_loss += loss[0].item()
                # total_loss += loss.item()
                num_batches += 1

                # Update progress bar
                pbar.set_postfix({
                    'Loss': f"{total_loss/num_batches:.6f}",
                    'LR': f"{self.optimizer.param_groups[0]['lr']:.1e}",
                    'f_px': f"{f_px:.2f}"
                })
                pbar.update(1)

                # Log images periodically
                if self.rank == 0 and batch_idx % 100 == 0:
                    
                    # plot org_image , original depth , predicted depth , canonical_inverse_depth , canonical_inverse_depth_pred in a grid 2x3
                    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

                    # plot original image
                    axs[0].imshow(batch['org_image'][0])
                    axs[0].set_title('Original Image')

                    # plot original depth
                    cax1 = axs[1].imshow(depth.squeeze().detach().cpu().numpy(), cmap='viridis')
                    axs[1].set_title('Original Depth')
                    fig.colorbar(cax1, ax=axs[1])

                    # plot predicted depth
                    cax2 = axs[2].imshow(preds['depth'].detach().cpu().numpy(), cmap='viridis')
                    axs[2].set_title('Predicted Depth')
                    fig.colorbar(cax2, ax=axs[2])

                    # plot loss components
                    ax = axs[3]
                    ax.bar(loss[1].keys(), loss[1].values(), color='skyblue')
                    ax.set_title('Loss Components')
                    ax.set_ylabel('Loss Value')
                    ax.set_xlabel('Loss Type')
                    ax.tick_params(axis='x', rotation=45)

                    plt.tight_layout()
                    plt.suptitle(f"{mode.capitalize()} Epoch {epoch} , f_px : {f_px:.2f}")

                    wandb.log({
                        "epoch": self.cur_epoch,
                        "image": wandb.Image(fig),
                        'mode': mode
                    })

                    plt.close()

            torch.cuda.empty_cache()

        return total_loss / num_batches

    def train_model(self): # NOTE : Update run both stages in for loop for stage in [stage1,stage2]
        """Execute complete two-stage training procedure."""
        try:
            # Stage 1: Training on real data
            self.setup_training_stage("stage1")
            logger.info("\nStarting training stage 1")
            self.data_setup(self.stages['stage1']['data_dir'], is_synthetic=False)

            if self.rank == 0:
                logger.info(f'Best Loss : {self.best_val_loss} before epoch : {self.cur_epoch}')

            for epoch in range(self.cur_epoch,self.stages['stage1']['epochs']):
                self.cur_epoch = epoch
                train_loss = self.train_epoch(epoch, mode='train')
                val_loss = self.train_epoch(epoch, mode='val')
                test_loss = self.test(epoch,'/home/badari/Thesis_Depth/Analysis/Test_Lab_Images/',[2,3,6,11,19,32])

                # combined_loss = self.get_combined_loss(val_loss, test_loss)

                self.scheduler.step(val_loss)
                # self.scheduler.step(combined_loss)
                
                if self.rank == 0:
                    self.log_metrics("stage1", epoch, train_loss, val_loss, test_loss)
                    self.save_checkpoint(epoch, train_loss, val_loss)
                    self.log_training_stats(epoch, len(self.train_loader) * epoch, train_loss)

                # Early stopping
                if val_loss < self.best_val_loss - self.min_delta:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                    
            # # Stage 2: Fine-tuning on synthetic data
            # self.setup_training_stage("stage2")
            # logger.info("\nStarting training stage 2")
            # start_epoch = self.stages['stage1']['epochs']
            # self.data_setup(self.stages['stage2']['data_dir'], is_synthetic=True)
            # for epoch in range(self.stages['stage2']['epochs']):
            #     self.cur_epoch = start_epoch + epoch
            #     train_loss = self.train_epoch(epoch, mode='train')
            #     val_loss = self.train_epoch(epoch, mode='val')
            #     test_loss = self.test(epoch,'/home/badari/Thesis_Depth/Analysis/Lab_Images/',[2,3,6,11,19,32])
                
            #     if self.rank == 0:
            #         self.log_metrics("stage2", epoch, train_loss, val_loss, test_loss)
            #         self.save_checkpoint(self.cur_epoch, train_loss, val_loss)
                
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

    @torch.no_grad()
    def test(self, epoch, PATH, idxs):
        """
        Perform inference and evaluation on test images.
        """
        self.model.eval()
        
        # Initialize metrics
        MAE_LOSS, MAGE_LOSS, TOTAL_LOSS = 0.0, 0.0, 0.0
        depth_metrics_avg = {'abs_rel': 0.0, 'rmse': 0.0, 'delta1': 0.0}
        boundary_metrics_avg = {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
        
        for idx in idxs:
            # Load data
            dep_path = os.path.join(PATH, f'depths/depth_data_{idx}.npy')
            img_path = os.path.join(PATH, f'images/color_image_{idx}.png')
            depth = np.load(dep_path)
            image = Image.open(img_path).convert('RGB')
            org_image = image.copy()

            image = self.transform(image)
            
            # Get prediction
            preds = self.infer(
                image.unsqueeze(0).to(self.device),
                f_px=torch.tensor(607.7325439453125).to(self.device)
            )
            
            pred_depth = preds['depth']
            fov_deg = preds['focallength_px']
            
            # Convert ground truth to tensor
            gt_depth = torch.tensor(depth).to(self.device)
            
            # Calculate losses
            loss = self.criterion(pred_depth.unsqueeze(0), gt_depth.unsqueeze(0).to(self.device))
            
            # Calculate additional metrics if rank is 0
            if self.rank == 0:
                # Depth metrics
                metrics = self.evaluate_depth_metrics(pred_depth, gt_depth, None)
                for k, v in metrics.items():
                    depth_metrics_avg[k] += v.item()
                
                # Boundary metrics 
                precision, recall = self.evaluate_boundaries(
                    pred_depth.cpu().numpy(), 
                    gt_depth.cpu().numpy()
                )
                boundary_metrics_avg['precision'] += precision
                boundary_metrics_avg['recall'] += recall
                boundary_metrics_avg['f1_score'] += 2 * (precision * recall) / (precision + recall + 1e-6)

            # Update loss totals
            MAE_LOSS += loss[1]['mae_loss']
            MAGE_LOSS += loss[1]['mage_loss']
            TOTAL_LOSS += loss[1]['total_loss']

            # Visualization code (existing plotting code stays the same)
            # Log to WandB
            if self.rank == 0:
                # plot and log
                fig,ax = plt.subplots(1,4,figsize=(20,5))

                # Original image
                ax[0].imshow(org_image)
                ax[0].set_title('Original Image')

                # Ground truth depth
                cax1 = ax[1].imshow(depth, cmap='viridis')
                ax[1].set_title('Ground Truth Depth')
                ax[1].figure.colorbar(cax1, ax=ax[1])

                # Predicted depth
                cax2 = ax[2].imshow(pred_depth.detach().cpu().numpy(), cmap='viridis')
                ax[2].set_title('Predicted Depth')
                ax[2].figure.colorbar(cax2, ax=ax[2])

                loss = self.criterion(pred_depth.unsqueeze(0), torch.tensor(depth).unsqueeze(0).to(self.device))

                # Create a bar plot for the losses
                ax[3].bar(loss[1].keys(), loss[1].values(), color='skyblue')
                ax[3].set_title('Loss Components')
                ax[3].set_ylabel('Loss Value')
                ax[3].set_xlabel('Loss Type')
                ax[3].tick_params(axis='x', rotation=45)

                plt.tight_layout()
                plt.suptitle(f'Epoch:{epoch}/Image:{idxs[0]}/fx:{fov_deg}/Loss:{loss[1]["mae_loss"]:.4f}')

                wandb.log({
                    "test_image": wandb.Image(fig),
                    "title": f'Epoch-{epoch}/Image-{idxs[0]}'
                })

        # Average the metrics
        num_images = len(idxs)
        MAE_LOSS /= num_images
        MAGE_LOSS /= num_images
        TOTAL_LOSS /= num_images
        
        if self.rank == 0:
            for k in depth_metrics_avg:
                depth_metrics_avg[k] /= num_images
            for k in boundary_metrics_avg:
                boundary_metrics_avg[k] /= num_images

            # Log all metrics to wandb
            wandb.log({
                "epoch": epoch,
                "mae_loss": MAE_LOSS,
                "mage_loss": MAGE_LOSS,
                "total_loss": TOTAL_LOSS,
                "depth_metrics/abs_rel": depth_metrics_avg['abs_rel'],
                "depth_metrics/rmse": depth_metrics_avg['rmse'],
                "depth_metrics/delta1": depth_metrics_avg['delta1'],
                "boundary_metrics/precision": boundary_metrics_avg['precision'],
                "boundary_metrics/recall": boundary_metrics_avg['recall'],
                "boundary_metrics/f1_score": boundary_metrics_avg['f1_score']
            })

            # Log metrics to console
            logger.info(f"Epoch {epoch} Test Metrics:")
            logger.info(f"Losses - MAE: {MAE_LOSS:.4f}, MAGE: {MAGE_LOSS:.4f}, Total: {TOTAL_LOSS:.4f}")
            logger.info(f"Depth Metrics - AbsRel: {depth_metrics_avg['abs_rel']:.4f}, "
                    f"RMSE: {depth_metrics_avg['rmse']:.4f}, Delta1: {depth_metrics_avg['delta1']:.4f}")
            logger.info(f"Boundary Metrics - Precision: {boundary_metrics_avg['precision']:.4f}, "
                    f"Recall: {boundary_metrics_avg['recall']:.4f}, F1: {boundary_metrics_avg['f1_score']:.4f}")

        return MAE_LOSS

    # f'Train-E_train:{self.cur_epoch} , W:{W} , f_px:{f_px}'
    def _log_images_(self,org_img,org_depth,pred_depth,title,mode):
        if self.rank == 0:
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))

            # plot original image
            axs[0].imshow(org_img[0])
            axs[0].set_title('Original Image')

            # plot original depth
            cax1 = axs[1].imshow(org_depth.detach().cpu().numpy(), cmap='viridis')
            axs[1].set_title('Original Depth')
            fig.colorbar(cax1, ax=axs[1])

            # plot predicted depth
            cax2 = axs[2].imshow(pred_depth.detach().cpu().numpy(), cmap='viridis')
            axs[2].set_title('Predicted Depth')
            fig.colorbar(cax2, ax=axs[2])

            plt.tight_layout()
            plt.suptitle(title)

            wandb.log({
                "epoch": self.cur_epoch,
                "image": wandb.Image(fig),
                'mode': 'train'
            })

            plt.close()
    
    def log_metrics(self, stage, epoch, train_loss, val_loss, test_loss=None):
        """Log training metrics to WandB."""
        if self.rank == 0:
            wandb.log({
                'epoch': epoch,
                f"{stage}_train_loss": train_loss,
                f"{stage}_val_loss": val_loss,
                f"{stage}_test_loss": test_loss,
            })
            logger.info(f"Epoch {epoch} - {stage} - Train Loss: {train_loss:.6f}, val Loss: {val_loss:.6f}, test Loss: {test_loss:.6f}")

    def get_combined_loss(self, val_loss, test_loss, alpha=0.8):
        """Weighted combination of validation and test loss"""
        return alpha * val_loss + (1 - alpha) * test_loss
    
    def evaluate_depth_metrics(self, pred_depth, gt_depth, valid_mask):
        """Implements comprehensive depth evaluation metrics from paper"""
        metrics = {}
        
        # AbsRel error
        metrics['abs_rel'] = torch.mean(torch.abs(pred_depth - gt_depth) / gt_depth)
        
        # RMSE
        metrics['rmse'] = torch.sqrt(torch.mean((pred_depth - gt_depth) ** 2))
        
        # δ1 threshold (% of pixels where max(pred/gt, gt/pred) < 1.25)
        thresh = torch.max(pred_depth/gt_depth, gt_depth/pred_depth)
        metrics['delta1'] = torch.mean((thresh < 1.25).float())
    
        return metrics

    def log_training_stats(self, epoch, step, train_loss):
        if self.rank == 0:
            wandb.log({
                'epoch': epoch,
                'step': step,
                'train_loss': train_loss,
                'encoder_lr': self.optimizer.param_groups[0]['lr'],
                'decoder_lr': self.optimizer.param_groups[1]['lr'],
                'lr_scale': self.get_lr_scale(step)
            })

    def save_checkpoint(self, epoch, train_loss, val_loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter,
            'current_stage': self.current_stage,
            'scaler_state_dict': self.scaler.state_dict()
        }
        
        save_path = f"checkpoints/{self.current_stage}_depth_pro_{epoch}.pt"
        torch.save(checkpoint, save_path)
        
        if val_loss < self.best_val_loss - self.min_delta:
            best_path = f"checkpoints/{self.current_stage}_depth_pro_best.pt"
            torch.save(checkpoint, best_path)
            self.best_val_loss = val_loss
            self.patience_counter = 0

    # def load_checkpoint(self, checkpoint_path):
    #     checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
    #     self.model.load_state_dict(checkpoint['model_state_dict'])
    #     self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    #     self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
    #     self.cur_epoch = checkpoint['epoch']
    #     self.best_val_loss = checkpoint['best_val_loss'] 
    #     self.patience_counter = checkpoint['patience_counter']
    #     self.current_stage = checkpoint['current_stage']
    
    def evaluate_boundaries(self, pred_depth, target_depth):
        """
        Evaluate boundary accuracy between predicted and target depth maps.
        
        Args:
            pred_depth: Predicted depth (numpy array or torch tensor)
            target_depth: Ground truth depth (numpy array or torch tensor)
        """
        # Convert to numpy if tensor
        if torch.is_tensor(pred_depth):
            pred_depth = pred_depth.cpu().numpy()
        if torch.is_tensor(target_depth):
            target_depth = target_depth.cpu().numpy()
        
        # Ensure the depth maps are uint8 for Canny
        pred_norm = cv2.normalize(pred_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        target_norm = cv2.normalize(target_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Apply Canny edge detection
        edges_pred = cv2.Canny(pred_norm, 100, 200)
        edges_target = cv2.Canny(target_norm, 100, 200)
        
        # Calculate metrics
        precision = np.sum(edges_pred & edges_target) / (np.sum(edges_pred) + 1e-6)
        recall = np.sum(edges_pred & edges_target) / (np.sum(edges_target) + 1e-6)
        
        return precision, recall


def main():
    # Set up distributed training
    world_size = torch.cuda.device_count()
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        mp.spawn(
            train_model,
            args=(world_size,),
            nprocs=world_size,
            join=True
        )
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    finally:
        cleanup()
        if wandb.run:
            wandb.finish()

def train_model(rank, world_size):
    setup(rank, world_size)
    trainer = DepthTrainer(rank, world_size)
    trainer.train_model()
    cleanup()

if __name__ == "__main__":
    main()