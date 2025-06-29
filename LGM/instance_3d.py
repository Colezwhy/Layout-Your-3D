
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

    # load mvdream for text-to-3D gaussians generation.
    pipe_text = MVDreamPipeline.from_pretrained(
    'ashawkey/mvdream-sd2.1-diffusers', # remote weights
    torch_dtype=torch.float16,
    trust_remote_code=True,
    # local_files_only=True,
    )
    pipe_text = pipe_text.to(device)
    
    # load imagedream for image-to-3D gaussians reconstruction
    pipe = MVDreamPipeline.from_pretrained(
        "ashawkey/imagedream-ipmv-diffusers", # remote weights
        torch_dtype=torch.float16,
        trust_remote_code=True,
        # local_files_only=True,
    )
    pipe = pipe.to(device) # MVDream pipeline

    # load rembg
    bg_remover = rembg.new_session()
    
    return model, bg_remover, pipe_text, pipe, proj_matrix

# process function, simply ancestral to the initialize_model function.
# TODO: add feature, reading the input images and then reconstruct with LGM. Not generating with MVDream.
def process(opt: Options, model, prompt, bg_remover, device, pipe, pipe_text, path, proj_matrix,
            input_elevation=0, input_num_steps=100, input_seed=24, is_t2i=True, input_multiview=None,
            prompt_neg='ugly, blurry, pixelated obscure, unnatural colors, poor lighting, dull, unclear, cropped, lowres, low quality, artifacts, duplicate',
            num=-1):
    
    kiui.seed_everything(input_seed)
    # Added for mention
    if is_t2i:
        if input_multiview is not None:
            print('Input text prompt, the multiview will not be used.')
    # Added for mention.
    
    name = os.path.splitext(os.path.basename(path))[0]
    print(f'[INFO] Processing {path} --> {name}')
    os.makedirs(opt.workspace, exist_ok=True)
    # text-conditioned
    if is_t2i:
        prompt_write = prompt.replace(' ', '_')
        # i.e. automatically get four camera view with elevation angles.
        mv_image_uint8 = pipe_text(prompt, negative_prompt=prompt_neg, num_inference_steps=input_num_steps, guidance_scale=7.5, elevation=input_elevation)
        mv_image_uint8 = (mv_image_uint8 * 255).astype(np.uint8)
        # bg removal
        mv_image = []
        saving_path = os.path.join(opt.workspace, name, prompt_write)
        Path(saving_path).mkdir(parents=True, exist_ok=True)
        for i in range(4):
            image = rembg.remove(mv_image_uint8[i], session=bg_remover) # [H, W, 4]
            im = numpy_to_pil(image)[0]
            # im.save(Path(os.path.join(saving_path, f'view{i}.png')))
            # to white bg
            image = image.astype(np.float32) / 255
            image = recenter(image, image[..., 0] > 0, border_ratio=0.2)
            image = image[..., :3] * image[..., -1:] + (1 - image[..., -1:])
            mv_image.append(image) # here restoring multi-view images.
    else:
        if input_multiview is None:
            saving_path = os.path.join(opt.workspace, opt.test_path)
            input_image = kiui.read_image(path, mode='uint8') # read into uint8 format.(255 based)
            # bg removal
            carved_image = rembg.remove(input_image, session=bg_remover) # [H, W, 4]
            mask = carved_image[..., -1] > 0

            # recenter
            image = recenter(carved_image, mask, border_ratio=0.2)
            
            # generate mv
            image = image.astype(np.float32) / 255.0

            # rgba to rgb white bg
            if image.shape[-1] == 4:
                image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])

            mv_image = pipe(prompt, image, negative_prompt=prompt_neg, num_inference_steps=input_num_steps, guidance_scale=5.0,  elevation=input_elevation)
            mv_image = np.stack([mv_image[1], mv_image[2], mv_image[3], mv_image[0]], axis=0) # [4, 256, 256, 3], float32
        else:
            # Support for SV3D generated instances.
            saving_path = os.path.dirname(input_multiview)
            name_ls_front = ['view_15.jpg', 'view_20.jpg', 'view_4.jpg', 'view_9.jpg']
            mv_image = []
            for name in name_ls_front: 
                pth_temp = os.path.join(input_multiview, name)
                input_image = kiui.read_image(pth_temp, mode='uint8')
                # input_image = np.array(im.resize((256, 256), resample = Image.LANCZOS)) #@NOTE: check if this will cause better reconstruction results?
                carved_image = rembg.remove(input_image, session=bg_remover) # [H, W, 4] 
                mask = carved_image[..., -1] > 0
                # recenter
                image = recenter(carved_image, mask, border_ratio=0.2)
                # res
                image = image.astype(np.float32) / 255
                # rgba to rgb white bg
                if image.shape[-1] == 4:
                    image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])
                mv_image.append(image)
                
            # for name in name_ls_front: 
            #     pth_temp = os.path.join(input_multiview, name)
            #     input_image = kiui.read_image(pth_temp, mode='uint8')
            #     # input_image = np.array(im.resize((256, 256), resample = Image.LANCZOS)) #@NOTE: check if this will cause better reconstruction results?
            #     carved_image = rembg.remove(input_image, session=bg_remover) # [H, W, 4] 
            #     mask = carved_image[..., -1] > 0
            #     # recenter
            #     image = recenter(carved_image, mask, border_ratio=0.2)
            #     # res
            #     image = image.astype(np.float32) / 255
            #     # rgba to rgb white bg
            #     if image.shape[-1] == 4:
            #         image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])
            #     mv_image_oblique.append(image)
            
    mv_image_grid = np.concatenate([
        np.concatenate([mv_image[1], mv_image[2]], axis=1),
        np.concatenate([mv_image[3], mv_image[0]], axis=1),
    ], axis=0)
    kiui.write_image(os.path.join(saving_path, 'multiview.jpg'), mv_image_grid)
    
    # Note that here we use mv_image variable for multi-view image storage (generated from MVDream pipeline.)
    # generate gaussians

    input_image = np.stack([mv_image[0], mv_image[1], mv_image[2], mv_image[3]], axis=0) # [4, 256, 256, 3], float32
    input_image = torch.from_numpy(input_image).permute(0, 3, 1, 2).float().to(device) # [4, 3, 256, 256]
    input_image = F.interpolate(input_image, size=(opt.input_size, opt.input_size), mode='bilinear', align_corners=False) # interpolate into input resolutions of LGM protocol.
    input_image = TF.normalize(input_image, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    
    # input_image_oblique = np.stack([mv_image[1], mv_image[2], mv_image[3], mv_image[0]], axis=0) # [4, 256, 256, 3], float32
    # input_image_oblique = torch.from_numpy(input_image_oblique).permute(0, 3, 1, 2).float().to(device) # [4, 3, 256, 256]
    # input_image_oblique = F.interpolate(input_image_oblique, size=(opt.input_size, opt.input_size), mode='bilinear', align_corners=False) # interpolate into input resolutions of LGM protocol.
    # input_image_oblique = TF.normalize(input_image_oblique, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    
    rays_embeddings = model.prepare_default_rays(device, azimuths=ray_frontal)
    input_image = torch.cat([input_image, rays_embeddings], dim=1).unsqueeze(0) # [1, 4, 9, H, W]
    # rays_embeddings_oblique = model.prepare_default_rays(device, azimuths=ray_oblique)
    # input_image_oblique = torch.cat([input_image_oblique, rays_embeddings_oblique], dim=1).unsqueeze(0) # [1, 4, 9, H, W]
    
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # forward inference pipeline
            gaussians = model.forward_gaussians(input_image) # here the generated gaussains have 256 ** 2 in number, which is not pruned with opacity.
            
        # save gaussians
        model.gs.save_ply(gaussians, os.path.join(saving_path, f'instance{num}.ply'))
        # gaussians = torch.cat([gaussians, gaussians_oblique], dim=1)

        # render 360 video 
        images = []
        elevation = 0

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
        imageio.mimwrite(os.path.join(saving_path, f'rendered_video_instance{num}.mp4'), images, fps=30)
    return mv_image

if __name__ == '__main__':
    
    opt = tyro.cli(AllConfigs)
    assert opt.test_path is not None
    if os.path.isdir(opt.test_path):
        file_paths = glob.glob(os.path.join(opt.test_path, "*"))
    else:
        file_paths = [opt.test_path]    
    
    # changing tyro will lead to memory leakage issue? Unknown problems, here we only define existing working_space and test_path
    instance_path = '../layout2image'
    instance_path = os.path.join(instance_path, opt.test_path)
    subfolders = [os.path.join(instance_path, name) for name in os.listdir(instance_path) if os.path.isdir(os.path.join(instance_path, name))] # Get all subfolders.
    file_paths = sorted([os.path.join(item, 'centralized_tiled.jpg') for item in subfolders])
    device = 'cuda:0'
    is_t2i = False
    prompt_set = ['A wooden cabinet', 'A TV']
    model, bg_remover, pipe_text, pipe, proj_matrix = initialize_model(opt, device)
    
    for i, path in enumerate(file_paths):
        process(opt, model, prompt_set[i], bg_remover, device, pipe, pipe_text, path, proj_matrix, is_t2i=is_t2i, num=i) # , input_multiview=input_multiview[i]