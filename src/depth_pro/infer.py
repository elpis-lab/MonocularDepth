import os
import sys
import logging
import tempfile
from datetime import timedelta
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import wandb

from src.depth_pro.network.encoder import DepthProEncoder
from src.depth_pro.network.decoder import MultiresConvDecoder
from src.depth_pro.training.dataset import DepthDataset
from src.depth_pro.training.losses import DepthLoss
from src.depth_pro import create_model_and_transforms
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import torchvision.transforms as transforms

from transformers import AutoModelForImageSegmentation
from torchvision import transforms

import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
birefnet = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True)
birefnet.to(device)

def extract_object(imagepath, birefnet=birefnet):
    # Data settings
    image_size = (1024, 1024)
    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(imagepath)
    org_image = image.copy()
    input_images = transform_image(image).unsqueeze(0).to('cuda')

    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)
    image.putalpha(mask)
    
    # Create background by removing object and filling with white
    background = org_image.copy()
    mask_array = np.array(mask)
    
    # Convert background to RGBA if it's not already
    if background.mode != 'RGBA':
        background = background.convert('RGBA')
    
    # Convert to numpy array for manipulation
    background_array = np.array(background)
    
    # Create white pixels (255, 255, 255, 255) where mask is high (object location)
    threshold = 128  # Adjust threshold as needed
    white_pixels = np.array([255, 255, 255, 255], dtype=np.uint8)
    background_array[mask_array > threshold] = white_pixels
    
    # Convert back to PIL Image
    background = Image.fromarray(background_array)
    # returns masked_image , mask , background
    return image, mask_array/255.0, background

def infer(model, img_path , org_depth_path=None, def_model=None , img_size=1536) :
    # load the image
    img = Image.open(img_path)
    img = img.convert("RGB")

    # extract object from image
    extracted_img, mask, background = extract_object(img_path)

    extracted_img = extracted_img.convert("RGB")

    # Initialize resizing transforms for PIL images
    resize_transform = transforms.Compose([
        transforms.Resize((img_size, img_size), 
                        interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor()  # Convert to tensor after resize
    ])

    # apply the resize transform
    img = resize_transform(img).unsqueeze(0).to(device)

    extracted_img = resize_transform(extracted_img).unsqueeze(0).to(device)

    # perform inference
    with torch.no_grad() :
        pred_depth = model.infer(img,f_px=torch.tensor(392.99432373046875).to(device))[0]
        pred_depth = 1.0 / torch.clamp(pred_depth, min=1e-6)
        print(f'Predicted Depth shape: {pred_depth.shape}')
        pred_depth = pred_depth.squeeze(0).cpu().numpy()
        print(f'Predicted Depth shape after squeeze: {pred_depth.shape}')

        e_pred_depth = model(extracted_img)[0]
        e_pred_depth = 1.0 / torch.clamp(e_pred_depth, min=1e-6)
        e_pred_depth = e_pred_depth.squeeze(0).cpu().numpy()

        if def_model is not None :
            def_pred_depth = def_model(img)[0]
            def_pred_depth = 1.0 / torch.clamp(def_pred_depth, min=1e-6)
            def_pred_depth = def_pred_depth.squeeze(0).cpu().numpy()

            e_def_pred_depth = def_model(extracted_img)[0]
            e_def_pred_depth = 1.0 / torch.clamp(e_def_pred_depth, min=1e-6)
            e_def_pred_depth = e_def_pred_depth.squeeze(0).cpu().numpy()

        else :
            def_pred_depth = np.zeros_like(pred_depth)
            e_def_pred_depth = np.zeros_like(e_pred_depth)

    # save the output
    os.makedirs(f'.outputs', exist_ok=True)
    save_path = f'.outputs/out_{os.path.basename(img_path)}'

    # load org_depth
    if org_depth_path is not None :
        org_depth = np.load(org_depth_path)
        org_depth = Image.fromarray(org_depth)

        # convert to pred_depth size and shape
        org_depth = org_depth.resize((pred_depth.shape[2], pred_depth.shape[1]))
        # org_depth = np.array([org_depth])

        # find rmse error b/w org_depth and pred_depth
        org_depth = np.array(org_depth)
        rmse = np.sqrt(np.mean((org_depth - pred_depth) ** 2))
        rmse_def = np.sqrt(np.mean((org_depth - def_pred_depth) ** 2))
        e_rmse = np.sqrt(np.mean((org_depth - e_pred_depth) ** 2))
        e_rmse_def = np.sqrt(np.mean((org_depth - e_def_pred_depth) ** 2))

        print(f"RMSE: {rmse} , Default RMSE: {rmse_def} , Extracted RMSE: {e_rmse} , Extracted Default RMSE: {e_rmse_def}")

        # Plot the results
        fig, ax = plt.subplots(1, 6, figsize=(24, 6))

        # Image plot
        ax[0].imshow(img.squeeze(0).permute(1, 2, 0).cpu().numpy())
        ax[0].set_title("Image")
        ax[0].axis("off")

        # Original Depth plot
        im1 = ax[1].imshow(org_depth, cmap="viridis")
        ax[1].set_title("Original Depth")
        ax[1].axis("off")
        plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

        # Predicted Depth plot
        im2 = ax[2].imshow(pred_depth.squeeze(0), cmap="viridis")
        ax[2].set_title("Predicted Depth")
        ax[2].axis("off")
        plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

        im3 = ax[3].imshow(def_pred_depth.squeeze(0), cmap="viridis")
        ax[3].set_title("Default Predicted Depth")
        ax[3].axis("off")
        plt.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.04)

        im4 = ax[4].imshow(e_pred_depth.squeeze(0), cmap="viridis")
        ax[4].set_title("Extracted Predicted Depth")
        ax[4].axis("off")
        plt.colorbar(im4, ax=ax[4], fraction=0.046, pad=0.04)

        im5 = ax[5].imshow(e_def_pred_depth.squeeze(0), cmap="viridis")
        ax[5].set_title("Extracted Default Predicted Depth")
        ax[5].axis("off")
        plt.colorbar(im5, ax=ax[5], fraction=0.046, pad=0.04)

        # Display the plot
        plt.suptitle(f"RMSE: {rmse:.4f} , Def RMSE : {rmse_def:.4f} , Extracted RMSE: {e_rmse:.4f} , Extracted Def RMSE: {e_rmse_def:.4f}")
        plt.tight_layout()
        # plt.show()

        plt.savefig(save_path)
    else :
        plt.imsave(save_path, pred_depth, cmap="gray")

    plt.close()

    # # print shapes of all
    # print(f"Original Depth Shape: {org_depth.shape}")
    # print(f"Predicted Depth Shape: {pred_depth.shape}")
    # print(f"Default Predicted Depth Shape: {def_pred_depth.shape}")

    pred_depth = pred_depth[0]
    def_pred_depth = def_pred_depth[0]
    e_pred_depth = e_pred_depth[0]
    e_def_pred_depth = e_def_pred_depth[0]

    print(f'Ori Depth shape: {org_depth.shape}')
    print(f'Predicted Depth shape after squeeze: {pred_depth.shape}')
    print(f'Default Predicted Depth shape after squeeze: {def_pred_depth.shape}')
    print(f'Extracted Predicted Depth shape after squeeze: {e_pred_depth.shape}')
    print(f'Extracted Default Predicted Depth shape after squeeze: {e_def_pred_depth.shape}')

    # get 10 random patches with random sizes
    for i in range(10) :

        print(f"Patch: {i+1}/{10} Processing...")

        x = np.random.randint(0, pred_depth.shape[1] - 128)
        y = np.random.randint(0, pred_depth.shape[0] - 128)
        w = np.random.randint(128, pred_depth.shape[1] - x)
        h = np.random.randint(128, pred_depth.shape[0] - y)

        # find rmse between org_depth and pred_depth
        if org_depth_path is not None :
            rmse = np.sqrt(np.mean((org_depth[y:y+h, x:x+w] - pred_depth[y:y+h, x:x+w]) ** 2))
            rmse_def = np.sqrt(np.mean((org_depth[y:y+h, x:x+w] - def_pred_depth[y:y+h, x:x+w]) ** 2))
            e_rmse = np.sqrt(np.mean((org_depth[y:y+h, x:x+w] - e_pred_depth[y:y+h, x:x+w]) ** 2))
            e_rmse_def = np.sqrt(np.mean((org_depth[y:y+h, x:x+w] - e_def_pred_depth[y:y+h, x:x+w]) ** 2))
        else :
            rmse = 0
            rmse_def = 0
            e_rmse = 0
            e_rmse_def = 0

        # plot the all 4 patches
        fig, ax = plt.subplots(1, 4, figsize=(16, 4))

        # Original Depth plot
        im1 = ax[0].imshow(org_depth[y:y+h, x:x+w], cmap="viridis")
        ax[0].set_title("Original Depth")
        ax[0].axis("off")
        plt.colorbar(im1, ax=ax[0], fraction=0.046, pad=0.04)

        # Predicted Depth plot
        im2 = ax[1].imshow(pred_depth[y:y+h, x:x+w], cmap="viridis")
        ax[1].set_title("Predicted Depth")
        ax[1].axis("off")
        plt.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04)

        im3 = ax[2].imshow(def_pred_depth[y:y+h, x:x+w], cmap="viridis")
        ax[2].set_title("Default Predicted Depth")
        ax[2].axis("off")
        plt.colorbar(im3, ax=ax[2], fraction=0.046, pad=0.04)

        im4 = ax[3].imshow(e_pred_depth[y:y+h, x:x+w], cmap="viridis")
        ax[3].set_title("Extracted Predicted Depth")
        ax[3].axis("off")
        plt.colorbar(im4, ax=ax[3], fraction=0.046, pad=0.04)

        # Display the plot
        plt.suptitle(f"RMSE: {rmse:.4f} , Def RMSE : {rmse_def:.4f} , Extracted RMSE: {e_rmse:.4f} , Extracted Def RMSE: {e_rmse_def:.4f}")

        plt.tight_layout()
        # plt.show()

        plt.savefig(f'.outputs/patch_{i}_{os.path.basename(img_path)}')

        plt.close()


if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description="Depth Pro Inference")

    # get paths for the checkpoint dir and the test images dir
    parser.add_argument("--checkpoint_dir", type=str, help="Path to the checkpoint directory")
    parser.add_argument("--test_images_dir", type=str, help="Path to the test images directory")
    parser.add_argument("--org_depths_path", type=str, help="Path to the original depth images directory")

    args = parser.parse_args()

    # get the checkpoint directory
    checkpoint_dir = args.checkpoint_dir
    test_images_dir = args.test_images_dir
    org_depths_path = args.org_depths_path

    # get the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the model
    model, _ = create_model_and_transforms(device=device, precision=torch.float32)

    def_checkpoint = torch.load('/home/badari/Thesis_Depth/Analysis/ml-depth-pro/checkpoints/depth_pro.pt', map_location=device, weights_only=True)

    def_model, _ = create_model_and_transforms(device=device, precision=torch.float32)

    def_model.load_state_dict(def_checkpoint)

    def_model.to(device)

    def_model.eval()

    # load the model
    model.load_state_dict(torch.load(checkpoint_dir, map_location=device, weights_only=True))#['model_state_dict']) 

    # move the model to the device
    model.to(device)

    # set the model to evaluation mode
    model.eval()

    infer(model, test_images_dir, org_depths_path, def_model)

    print("Inference Done... !!!")
    
""" Inference Command
CUDA_VISIBLE_DEVICES=1 python src/depth_pro/infer.py --checkpoint_dir /home/badari/Thesis_Depth/Analysis/ml-depth-pro/src/depth_pro/checkpoints/stage1_depth_pro_19.pt --test_images_dir /home/badari/Thesis_Depth/Analysis/Test_Images/color_image_2.png --org_depths_path /home/badari/Thesis_Depth/Analysis/Test_Images/depth_data_2.npy
"""