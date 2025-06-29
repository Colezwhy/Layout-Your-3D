import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel, AutoImageProcessor, AutoModel
import torch.nn as nn
import time
from instance_3d import initialize_model
from core.options import AllConfigs, Options
import tyro
import numpy as np
import tqdm
from kiui.cam import orbit_camera
import kiui
import os
from pathlib import Path

def get_rotation_matrix(axis, theta):
    """
    Return rotation matrix
    :param axis: rotate around the axis
    :param theta: ratation angle
    :return
    """
    if axis == 'x':
        return np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])
    elif axis == 'y':
        return np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
    elif axis == 'z':
        return np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Invalid rotation axis. Choose from 'x', 'y', or 'z'.")
    

def rotate_points(points, axis, angle_degrees):
    """
    Get ratation for each point
    :param points: point cloud dat
    :param axis: axis that needs to be rotated around.
    :param angle_degrees: rotation angle
    :return
    """
    theta = np.radians(angle_degrees) 
    rotation_matrix = get_rotation_matrix(axis, theta) 
    rotation_matrix = torch.from_numpy(rotation_matrix).to('cuda').to(torch.float32)
    points = points[0]
    points[..., :3] = torch.matmul(points[..., :3], rotation_matrix.T)
    rotated_points = torch.unsqueeze(points, dim=0)
    
    
    return rotated_points


if __name__ == '__main__':
    
    opt = tyro.cli(AllConfigs)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model_lgm, bg_remover, pipe_text, pipe, proj_matrix = initialize_model(opt, device)
    asset = 'fire_hydrant'
    ins = -1
    
    axis = 'x'
    instance_storage = f'./LGM/3d_instance_storage/{asset}'
    instances_3d = sorted(os.listdir(instance_storage))
    
    for i, instance in enumerate(instances_3d):
        if 'instance' not in instance:
            continue
        if 'ply' not in instance:
            continue
        if 'rectified' in instance:
            continue
        ins = ins + 1
        
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                # forward inference pipeline
                gaussians = model_lgm.gs.load_ply(os.path.join(instance_storage, instance)) # here the generated gaussains have 256 ** 2 in number, which is not pruned with opacity.
            # gaussians = torch.cat([gaussians, gaussians_oblique], dim=1)

            # render 360 video 
            images = []
            elevation = 0
            azimuth = np.arange(0, 360, 10, dtype=np.int32)
            for azi in tqdm.tqdm(azimuth):
                
                cam_poses = torch.from_numpy(orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)

                cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
                
                # cameras needed by gaussian rasterizer
                cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
                cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
                cam_pos = - cam_poses[:, :3, 3] # [V, 3]

                image = model_lgm.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=1)['image']
                images.append((image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))

            images = np.concatenate(images, axis=0)
            # imageio.mimwrite(os.path.join(saving_path, f'rendered_video_instance{num}.mp4'), images, fps=30)

        pil_images = [Image.fromarray(img) for img in images]
        # processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        # model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)

        start_total_time = time.time()

        start_time = time.time()
        image1 = Image.open(f'./layout2image/{asset}/instance{str(ins)}/centralized_tiled.jpg')
        with torch.no_grad():
            # inputs1 = processor(images=image1, return_tensors="pt").to(device)
            # image_features_ref = model.get_image_features(**inputs1)
            
            # ======================= DINOV2 ========================= #
            inputs1 = processor(images=image1, return_tensors="pt").to(device)
            outputs1 = model(**inputs1)
            image_features_ref = outputs1.last_hidden_state
            image_features_ref = image_features_ref.mean(dim=1)
            # ======================= DINOV2 ========================= #
        end_time = time.time()
        print(f"ref image time cost: {end_time - start_time:.4f}seconds")

        # rendered list
        start_time = time.time()
        feats_render = []
        with torch.no_grad():
            for item in pil_images:
                # ======================= CLIP ========================= #
                # input = processor(images=item, return_tensors="pt").to(device)
                # image_features = model.get_image_features(**input)
                # ======================= CLIP ========================= #
                
                # ======================= DINOV2 ========================= #
                input = processor(images=item, return_tensors="pt").to(device)
                outputs1 = model(**input)
                image_features = outputs1.last_hidden_state
                image_features = image_features.mean(dim=1)
                # ======================= DINOV2 ========================= #
                
                feats_render.append(image_features)
        end_time = time.time()
        print(f"rendered images time cost: {end_time - start_time:.4f}seconds")

        # 计算余弦相似度
        start_time = time.time()
        cos = nn.CosineSimilarity(dim=0)
        sim_list = []
        for ft in feats_render:
            sim = cos(image_features_ref[0], ft[0]).item()
            sim = (sim + 1) / 2
            sim_list.append(sim)
            
        sim_list = np.array(sim_list)
        end_time = time.time()
        print(f"Cos similarity time cost: {end_time - start_time:.4f}seconds")

        # 打印总时间
        end_total_time = time.time()
        print(f"Total time: {end_total_time - start_total_time:.4f}seconds")

        best_idx = np.argmax(sim_list)
        best = sim_list[best_idx]
        
        # saving rotated gaussians.
        theta = - azimuth[best_idx]

        if ins == 1:
            theta = -12.5
        print(theta)
        pointcloud = rotate_points(gaussians, axis, theta) # rotate around the y axis, the azimuth being theta.
        
        kiui.write_image(os.path.join(f'./LGM/3d_instance_storage/{asset}', f'best{str(ins)}.jpg'), images[best_idx])
        Path(f'./LGM/3d_instance_storage/{asset}/example').mkdir(exist_ok=True, parents=True)
        # for i, item in enumerate(images):
        #     kiui.write_image(os.path.join(f'./LGM/3d_instance_storage/{asset}/example', f'example{i}.jpg'), item)  
        print('Similarity:', best)
        model_lgm.gs.save_ply(pointcloud, os.path.join(f'./LGM/3d_instance_storage/{asset}/instance{str(ins)}_rectified.ply'))
        
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                # forward inference pipeline
                gaussians_refined = model_lgm.gs.load_ply(f'./LGM/example_fin/{asset}/instance{ins}.ply') ## 
        pointcloud_refined = rotate_points(gaussians_refined, axis, theta)
        model_lgm.gs.save_ply(pointcloud_refined, os.path.join(f'./LGM/example_fin/{asset}/instance{ins}_rectified.ply'))