
import os
import tyro
import glob
import imageio
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from safetensors.torch import load_file
import rembg
import argparse
from diffusers.utils import numpy_to_pil
from PIL import Image

from pathlib import Path
import kiui
from kiui.op import recenter
from kiui.cam import orbit_camera

from core.options import AllConfigs, Options
from core.models import LGM
# from mvdream.pipeline_mvdream import MVDreamPipeline
from mvdream.pipeline_mvdream import MVDreamPipeline

vid_degrees = np.array([17.14285714, 34.28571429, 51.42857143, 68.57142857, 85.71428571, 
 102.85714286, 120.0, 137.14285714, 154.28571429, 171.42857143, 
 188.57142857, 205.71428571, 222.85714286, 240.0, 257.14285714,
 274.28571429, 291.42857143, 308.57142857, 325.71428571, 342.85714286, 0.])

ray_frontal = [0, 90, 180, 270]
ray_oblique = [45, 135, 225, 315]

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

'''
This script for image-to-3D for a single instance within a composed prompt.
modified from SV3D.
'''

def initialize_model(opt, device):
    # initialize model
    model = LGM(opt)
    # resume pretrained checkpoint
    if opt.resume is not None:
        if opt.resume.endswith('safetensors'):
            ckpt = load_file(opt.resume, device='cpu')
        else:
            ckpt = torch.load(opt.resume, map_location='cpu')
        model.load_state_dict(ckpt, strict=False)
        print(f'[INFO] Loaded checkpoint from {opt.resume}')
    else:
        print(f'[WARN] model randomly initialized, are you sure?')

    # device, from input initialization
    model = model.half().to(device)
    model.eval()

    tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy)) # Get half angle of fov and do a tangent conversion.
    proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
    proj_matrix[0, 0] = 1 / tan_half_fov
    proj_matrix[1, 1] = 1 / tan_half_fov
    proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
    proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
    proj_matrix[2, 3] = 1
    
    return model, proj_matrix

# process function, simply ancestral to the initialize_model function.
# TODO: add feature, reading the input images and then reconstruct with LGM. Not generating with MVDream.
def process(opt: Options, model, device, proj_matrix, num=-1):
    saving_path = './MIGC/4D_trial_images'
    with torch.no_grad():
        # save gaussians
        gaussians = model.gs.load_ply('./LGM/temp_storage/stroller_1_rec.ply')# os.path.join(saving_path, 'instance1.ply'))
        # gaussians = torch.cat([gaussians, gaussians_oblique], dim=1)

        # render 360 video 
        images = []
        elevation = 0

        if opt.fancy_video: # default as saving
            azimuth = np.arange(0, 720, 4, dtype=np.int32)
            for azi in tqdm.tqdm(azimuth):
                
                cam_poses = torch.from_numpy(orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)

                cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
                
                # cameras needed by gaussian rasterizer
                cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
                cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
                cam_pos = - cam_poses[:, :3, 3] # [V, 3]

                scale = min(azi / 360, 1)

                image = model.gs.render(gaussians, cam_view.unsqueeze, cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=scale)['image']
                images.append((image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))
        else:
            azimuth = np.arange(0, 360, 2, dtype=np.int32)
            for azi in tqdm.tqdm(azimuth):
                
                cam_poses = torch.from_numpy(orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)

                cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
                
                # cameras needed by gaussian rasterizer
                cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
                cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
                cam_pos = - cam_poses[:, :3, 3] # [V, 3]

                image = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=1)['image']
                images.append((image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))

        images = np.concatenate(images, axis=0)
        imageio.imwrite(os.path.join(saving_path, f'view.png'), images[0])
        imageio.mimwrite(os.path.join(saving_path, f'rendered_video_compositional.mp4'), images, fps=30)
    return

if __name__ == '__main__':
    
    opt = tyro.cli(AllConfigs)
    # changing tyro will lead to memory leakage issue? Unknown problems, here we only define existing working_space and test_path
    device = 'cuda:0'
    model, proj_matrix = initialize_model(opt, device)    
    process(opt, model, device, proj_matrix) # , input_multiview=input_multiview[i]