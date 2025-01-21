import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import logging
from PIL import Image
import glob
import os
import torchvision.transforms as transforms
from torchvision.transforms import (
    Compose,
    ConvertImageDtype,
    Lambda,
    Normalize,
    ToTensor,
)
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
from typing import Dict, Tuple
import cv2

logger = logging.getLogger(__name__)

class DepthAugmenter:
    def __init__(self, size=(1536, 1536), color_prob=0.75, blur_prob=0.30, crop_prob=0.50, is_synthetic=False):
        self.size = size
        self.is_synthetic = is_synthetic
        
        # Adjust probabilities based on stage
        self.color_prob = color_prob * (0.7 if is_synthetic else 1.0)
        self.blur_prob = blur_prob * (0.7 if is_synthetic else 1.0)
        self.crop_prob = crop_prob * (0.6 if is_synthetic else 1.0)
        
        # Initialize transforms
        self.color_transform = T.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1
        )
        
        self.blur_kernels = [3, 5, 7]
        self.resize = T.Resize(self.size, antialias=True)

    def apply_color_jitter(self, image: torch.Tensor) -> torch.Tensor:
        """Apply color jittering with probability"""
        try:
            if torch.rand(1) < self.color_prob:
                if len(image.shape) == 3:
                    c, h, w = image.shape
                    image = image.unsqueeze(0)
                image = self.color_transform(image)
                image = image.squeeze(0)
        except Exception as e:
            logger.error(f"Error in color jitter: {str(e)}")
        return image

    def apply_blur(self, image: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian blur with probability"""
        try:
            if torch.rand(1) < self.blur_prob:
                kernel_size = np.random.choice(self.blur_kernels)
                blur = T.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0))
                image = blur(image)
        except Exception as e:
            logger.error(f"Error in blur: {str(e)}")
        return image

    def safe_resize(self, image: torch.Tensor, depth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Safely resize both image and depth to target size"""
        try:
            # Handle image
            # if len(image.shape) == 3:
            #     image = image.unsqueeze(0)
            # image = self.resize(image).squeeze(0)
            
            # # Handle depth
            # if len(depth.shape) == 2:
            #     depth = depth.unsqueeze(0).unsqueeze(0)
            # depth = self.resize(depth).squeeze(0).squeeze(0)
            
            return image, depth
        except Exception as e:
            logger.error(f"Error in resize: {str(e)}")
            return image, depth

    def apply_augmentations(self, image: torch.Tensor, depth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply all augmentations in sequence"""
        try:
            # Color augmentations (only to image)
            image = self.apply_color_jitter(image)
            image = self.apply_blur(image)
            
            # First resize both to target size
            # image, depth = self.safe_resize(image, depth)
            
            # Return augmented tensors
            return image, depth
            
        except Exception as e:
            logger.error(f"Error in augmentations: {str(e)}")
            # Fallback: just resize
            return self.safe_resize(image, depth)

    def __call__(self, image: torch.Tensor, depth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Main entry point for augmentations"""
        try:
            return self.apply_augmentations(image, depth)
        except Exception as e:
            logger.error(f"Error in augmenter call: {str(e)}")
            # Always return valid tensors
            return self.safe_resize(image, depth)

class DepthDataset(Dataset):
    def __init__(self, data_dir, img_size=1536, is_train=True, is_synthetic=False):
        """Initialize dataset with stage-specific settings."""
        super().__init__()
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.is_train = is_train
        self.is_synthetic = is_synthetic
        
        # Constants
        self.min_valid_depth = 0.01
        self.max_valid_depth = 10.0
        
        # Initialize augmenter based on stage
        if self.is_train:
            self.augmenter = DepthAugmenter(
                size=(img_size, img_size),
                color_prob=0.9 if not is_synthetic else 0.5,
                blur_prob=0.50 if not is_synthetic else 0.2,
                crop_prob=0.50 if not is_synthetic else 0.3,
                is_synthetic=is_synthetic
            )
        else:
            self.augmenter = None
            
        # Base transforms
        self.normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        self.resize_transform = transforms.Compose([
            # transforms.Resize((img_size, img_size), 
            #                 interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

        self.F_PX = {"640":607.7}
        
        # Setup paths and file lists
        self._setup_file_lists()
        
        logger.info(f"Initialized {'Synthetic' if is_synthetic else 'Real'} dataset with {len(self)} samples")

    def _setup_file_lists(self):
        """Setup image and depth file lists."""
        image_path = self.data_dir / "images"
        depth_path = self.data_dir / "depths"
        
        if not image_path.exists() or not depth_path.exists():
            raise ValueError(f"Missing directories: {image_path} or {depth_path}")
        
        # Get all image files
        image_files = sorted(glob.glob(str(image_path / "*.png")))
        
        self.image_files = []
        self.depth_files = []
        
        for img_file in image_files:
            img_name = Path(img_file).stem  
            depth_file = depth_path / f"{img_name}.png"  # Assuming .npy depth files
            
            if depth_file.exists():
                self.image_files.append(img_file)
                self.depth_files.append(str(depth_file))

        if len(self.image_files) == 0:
            raise ValueError(f"No matching image and depth files found in {self.data_dir}")
        
        logger.info(f"Found {len(self.image_files)} {'synthetic' if self.is_synthetic else 'real'} image-depth pairs")

    def preprocess_image(self, image):
        """Apply stage-specific image preprocessing."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        # Apply base transforms
        image = self.resize_transform(image)
        
        # # Apply augmentations if in training mode
        # if self.is_train and self.augmenter is not None:
        #     image = self.augmenter.apply_color_jitter(image)
        #     image = self.augmenter.apply_blur(image)
            
        # Normalize
        image = self.normalize(image)
        return image

    def preprocess_depth(self, depth):
        """Apply stage-specific depth preprocessing."""
        if isinstance(depth, np.ndarray):
            depth = torch.from_numpy(depth)
            
        # Convert to float32 and normalize
        depth = depth.float()
        if self.is_synthetic:
            depth = depth / 10000.0  # Convert mm to meters for synthetic data
        else:
            depth = depth / 10000.0  # Convert 1e-4 scale to meters for real data
            
        # Clip depth values
        depth = torch.clamp(depth, self.min_valid_depth, self.max_valid_depth)
        return depth

    def __getitem__(self, idx):
        try:
            # Load image
            with Image.open(self.image_files[idx]) as img:
                image = img.convert('RGB')
                org_image = np.array(image)
                H, W = org_image.shape[:2]
                
            # Load depth
            # depth = np.load(self.depth_files[idx], allow_pickle=True).astype(np.float32)
            with Image.open(self.depth_files[idx]) as depth_img:
                depth = np.array(depth_img, dtype=np.float32)
                # resize the depth image to the same size as the image
                depth = nn.functional.interpolate(torch.tensor(depth).unsqueeze(0).unsqueeze(0), size=(H, W), mode='nearest').squeeze(0).squeeze(0).numpy()
                depth = self.preprocess_depth(depth)
            # H, W = depth.shape
            
            # Convert to tensors
            image = self.preprocess_image(image)
            
            # # Apply augmentations if in training mode
            if self.is_train and self.augmenter is not None:
                image, depth = self.augmenter(image, depth)
            # else:
            # Just resize if not training

                    
                # image = T.Resize(self.img_size, antialias=True)(image).squeeze(0)
                # depth = T.Resize(self.img_size, antialias=True)(depth).squeeze(0).squeeze(0)
            
            # find image size
            # _, H, W = image.shape

            # # Get focal length
            # focal_length = self.F_PX[str(W)] if str(W) in self.F_PX else None
            # if focal_length is not None:
            #     focal_length = self.scale_focal_length(focal_length, W, self.img_size)
            #     focal_length = torch.tensor(focal_length, dtype=torch.float32)
            # else:
            #     focal_length = torch.tensor(0.0, dtype=torch.float32)  # Default value
            # focal_length = torch.tensor(0.0, dtype=torch.float32)  # Default value
            
            # Ensure all returned values are valid tensors
            return {
                "org_image": torch.from_numpy(org_image),
                "image": image,
                "depth": depth,
                "W": torch.tensor(W, dtype=torch.int32),
                # "focal_length": focal_length,
                "is_synthetic": torch.tensor(self.is_synthetic, dtype=torch.bool)
            }
                
        except Exception as e:
            logger.error(f"Error loading data at index {idx}")
            logger.error(f"Image: {self.image_files[idx]}")
            logger.error(f"Depth: {self.depth_files[idx]}")
            logger.error(f"Error: {str(e)}")
            raise

    def __len__(self):
        return len(self.image_files)

    @staticmethod
    def scale_focal_length(f_px, original_size, new_size):
        """Scale focal length based on resize ratio."""
        scale_factor = new_size / original_size
        return f_px * scale_factor