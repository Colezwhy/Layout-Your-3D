import os
from dataclasses import dataclass, field
import numpy as np
import random
import threestudio
import torch
from threestudio.systems.base import BaseLift3DSystem
from threestudio.systems.utils import parse_optimizer, parse_scheduler
from threestudio.utils.loss import tv_loss
from threestudio.utils.typing import *
from kiui.lpips import LPIPS
from ..utils.layout_utils import ratio, transform_gaussians, load_ply, save_ply, initialize_3d_layouts
from transformers import AutoImageProcessor, AutoModel, AutoConfig, Dinov2Model
from PIL import Image
import math
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)

from torch_kdtree import build_kd_tree

# from ..geometry.gaussian_base import BasicPointCloud

'''
Here we assume the given condition are 2D layout boxes, and given in the CONFIG files. The layout is loaded in the initialization process.
Also we here apply the transformation for initialized 3D layout in this file for better explanability.
After initialization, the initialized 3D boxes are then transformed into optimizable parameters that controls the relative positions of given box.
NOTE@: Now we only consider depth and rotation as differentiable parameters that could be optimized.
TODO@: Implementing loading Layouy 2D boxes and then transform into 3D boxes.
        CUSTOMIZED PARAMETERS:
        self.geometry.layout_2d: numpy->ndarray, the user given 2D layout boxes.
        self.geometry.box_configs: initialized 3D layout box.
        self.geometry.scaling_factors: for rescale
        self.geometry.mark_instances: for marking how many gaussians are there representing each instance.
HERE WE MENTION THAT: at the end of each iteration, we should update the given xyz.
'''

@threestudio.register("gaussian-splatting-layout-system")
class LayoutGSSystem(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        visualize_samples: bool = False
        
    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        # self.pre_process_instances()
        super().configure()
        self.automatic_optimization = False
        # self.lpips = LPIPS(net='vgg').to(self.device)
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.prompt_utils = self.prompt_processor()
        
        #  edited by colez
        
        if self.cfg.loss['lambda_feat_recon_loss'] > 0.0 and self.geometry.cfg.optimize_layout:
            self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
            self.feat_model = AutoModel.from_pretrained("facebook/dinov2-base").to('cuda')
            # frontal view batch preparation
            self.frontal_batch = self.get_frontal_camera_batch()
            # Prepare embedding     
            image_ref = Image.open(f'../MIGC/layout2image/{os.path.basename(self.geometry.cfg.geometry_convert_from)}/foreground_composed.jpg')
            inputs1 = self.processor(images=image_ref, return_tensors="pt").to('cuda')
            outputs1 = self.feat_model(**inputs1)
            self.image_features_ref = outputs1.last_hidden_state[0, 1 : , ...]
            self.L2_loss = nn.MSELoss()
            
        if self.cfg.loss['lambda_collision'] > 0.0 and self.geometry.cfg.optimize_layout:
            self.mark_accum = torch.cumsum(self.geometry.mark_instances, dim=0)
            disparse_lst = []
            means_lst = []
            for item in range(self.geometry.num_layouts):
                if item == 0:
                    pc = self.geometry.get_xyz[0:self.mark_accum[item]]
                else:
                    pc = self.geometry.get_xyz[self.mark_accum[item-1]:self.mark_accum[item]]
                pc_mean = torch.mean(pc, dim=0)
                means_lst.append(pc_mean)
                distance_tensor = torch.norm(pc - pc_mean, dim=-1, keepdim=True)
                disp = torch.mean(distance_tensor)
                disparse_lst.append(disp)

            self.disparity_lst = disparse_lst # the mean distance of gaussians to each instance
    
    def calculate_collision(self, pc1, pc2):
        """
        point cloud distance
        theta: float, threshold for distinguishing the loss.
        """
        # KDTree cuda version
        tree1 = build_kd_tree(pc1)
        tree2 = build_kd_tree(pc2)

        # Check the nearest neighbour(1)
        distances1, ind1 = tree1.query(pc2, nr_nns_searches=1)
        print(distances1)
        distances2, ind2 = tree2.query(pc1, nr_nns_searches=1)

        return distances1, distances2, ind1, ind2
    
    def get_frontal_camera_batch(self):
        # customization of different parameters.
        batch_size = 1
        height: float = 224
        light_distance: float = 1.0
        # customization of different parameters.
        # By Colez.
        elevation_deg: Float[Tensor, "B"]
        elevation: Float[Tensor, "B"]
        elevation = torch.zeros(1).to('cuda')
        elevation_deg = elevation * math.pi / 180

        azimuth_deg: Float[Tensor, "B"]
        azimuth_deg =  torch.zeros(1).to('cuda')
        azimuth = azimuth_deg * math.pi / 180

        # sample distances from a uniform distribution bounded by distance_range
        camera_distances: Float[Tensor, "B"] = (torch.tensor([1.5]).to('cuda'))

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,)

        # Not changed.
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions).to('cuda')
        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(batch_size, 1).to('cuda')
        
        # fixed fovy 
        fovy_deg: Float[Tensor, "B"] = (torch.tensor([49.1]).to('cuda'))
        fovy = fovy_deg * math.pi / 180

        # here make this function static
        light_distances: Float[Tensor, "B"] = (
            torch.tensor([light_distance]).to('cuda')
        )

        # do not apply light perturb
        light_direction: Float[Tensor, "B 3"] = F.normalize(camera_positions, dim=-1)
        # get light position by scaling light direction by light distance
        light_positions: Float[Tensor, "B 3"] = (
            light_direction * light_distances[:, None]
        )

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1).to('cuda')
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        # get directions by dividing directions_unit_focal by focal length
        
        directions_unit_focal = get_ray_directions(H=height, W=height, focal=1.0)
        
        focal_length: Float[Tensor, "B"] = 0.5 * height / torch.tan(0.5 * fovy)
        directions: Float[Tensor, "B H W 3"] = directions_unit_focal[None, :, :, :].repeat(batch_size, 1, 1, 1).to('cuda')
        directions[:, :, :, :2] = (directions[:, :, :, :2] / focal_length[:, None, None, None])

        # Importance note: the returned rays_d MUST be normalized!
        rays_o, rays_d = get_rays(
            directions, c2w, keepdim=True, normalize=False
        )

        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy, height / height, 0.01, 100.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx.to('cuda'))

        return {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "mvp_mtx": mvp_mtx,
            "camera_positions": camera_positions,
            "c2w": c2w,
            "light_positions": light_positions,
            "elevation": elevation_deg,
            "azimuth": azimuth_deg,
            "camera_distances": camera_distances,
            "height": height,
            "width": height,
            "fovy": fovy,
            "proj_mtx": proj_mtx,
        }
        
    def configure_optimizers(self):
        optim = self.geometry.optimizer
        if hasattr(self, "merged_optimizer"):
            return [optim]
        if hasattr(self.cfg.optimizer, "name"):
            net_optim = parse_optimizer(self.cfg.optimizer, self)
            optim = self.geometry.merge_optimizer(net_optim)
            self.merged_optimizer = True
        else:
            self.merged_optimizer = False
        return [optim]

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        self.geometry.update_learning_rate(self.global_step)
        outputs = self.renderer.batch_forward(batch)
        return outputs

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        out = self(batch) # outputs.
        visibility_filter = out["visibility_filter"]
        radii = out["radii"]
        guidance_inp = out["comp_rgb"]
        viewspace_point_tensor = out["viewspace_points"]
        guidance_out = self.guidance(
            guidance_inp,
            self.prompt_utils,
            **batch,
            rgb_as_latents=False
        )
        
        loss_sds = 0.0
        loss = 0.0

        self.log(
            "gauss_num",
            int(self.geometry.get_xyz.shape[0]),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss_sds += value * self.C(
                    self.cfg.loss[name.replace("loss_", "lambda_")]
                )

        # gaussian related losses
        
        xyz_mean = None
        if self.cfg.loss["lambda_position"] > 0.0:
            xyz_mean = self.geometry.get_xyz.norm(dim=-1)
            loss_position = xyz_mean.mean()
            self.log(f"train/loss_position", loss_position)
            loss += self.C(self.cfg.loss["lambda_position"]) * loss_position

        if self.cfg.loss["lambda_opacity"] > 0.0:
            scaling = self.geometry.get_scaling.norm(dim=-1)
            loss_opacity = (
                scaling.detach().unsqueeze(-1) * self.geometry.get_opacity
            ).sum()
            self.log(f"train/loss_opacity", loss_opacity)
            loss += self.C(self.cfg.loss["lambda_opacity"]) * loss_opacity

        if self.cfg.loss["lambda_sparsity"] > 0.0:
            loss_sparsity = -(self.geometry.get_opacity - 0.5).pow(2).mean()
            self.log("train/loss_sparsity", loss_sparsity)
            loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)
            
        if self.cfg.loss["lambda_scales"] > 0.0:
            scale_sum = torch.sum(self.geometry.get_scaling)
            self.log(f"train/scales", scale_sum)
            loss += self.C(self.cfg.loss["lambda_scales"]) * scale_sum

        if self.cfg.loss["lambda_tv_loss"] > 0.0:
            loss_tv = self.C(self.cfg.loss["lambda_tv_loss"]) * tv_loss(
                out["comp_rgb"].permute(0, 3, 1, 2)
            )
            self.log(f"train/loss_tv", loss_tv)
            loss += loss_tv
            
        # =========== Normal losses =========== #

        if (
            out.__contains__("comp_depth")
            and self.cfg.loss["lambda_depth_tv_loss"] > 0.0
        ):
            loss_depth_tv = self.C(self.cfg.loss["lambda_depth_tv_loss"]) * (
                tv_loss(out["comp_normal"].permute(0, 3, 1, 2))
                + tv_loss(out["comp_depth"].permute(0, 3, 1, 2))
            )
            self.log(f"train/loss_depth_tv", loss_depth_tv)
            loss += loss_depth_tv
            
        if self.cfg.loss["lambda_normal_smooth_loss"] > 0.0:
            if "comp_normal" not in out:
                raise ValueError(
                    "comp_normal is required for 2D normal smooth loss, no comp_normal is found in the output."
                )
            normal = out["comp_normal"]
            loss_normal_smooth = self.C(self.cfg.loss["lambda_normal_smooth_loss"]) * ((
                normal[:, 1:, :, :] - normal[:, :-1, :, :]).square().mean() \
                + (normal[:, :, 1:, :] - normal[:, :, :-1, :]).square().mean())
            self.log(f"train/loss_normal_smooth", loss_normal_smooth)
            loss += loss_normal_smooth
            
        # =========== Normal losses =========== #
        
        
        # =========== Layout refinement losses =========== #
        loss_layout = 0.0
        loss_ref = 0.0
        loss_collision = 0.0
        # reference loss
        if self.cfg.loss['lambda_feat_recon_loss'] > 0.0:
            out_frontal = self(self.frontal_batch)
            front_view = out_frontal["comp_rgb"] # in torch format
            # front_mask = out_frontal["mask"]
            
            sav = (front_view * 256)[0].detach().cpu().numpy().astype(np.uint8)
            pil_image = Image.fromarray(sav)
            pil_image.save('../3Dgen/threestudio/custom/threestudio-3dgs/render.png')
            
            with torch.no_grad(): #disable DINOv2 update
                outputs2 = self.feat_model(pixel_values=front_view.permute(0, 3, 1, 2)) 
            image_features_rendered = outputs2.last_hidden_state[0, 1 : , ...]
            # Here apply feature level L2 loss, mean
            loss_ref = self.L2_loss(image_features_rendered.view(-1), self.image_features_ref.view(-1)) * self.cfg.loss['lambda_feat_recon_loss']
            self.log(f"train/loss_reference", loss_ref)
            loss_layout += loss_ref
        
        
        # collision loss
        if self.cfg.loss['lambda_collision'] > 0.0 and self.global_step % 5 == 0:
            samples = list(range(self.geometry.num_layouts))
            selected_instance_nums = random.sample(samples, 2)
            
            means3D = self.geometry.get_xyz * 1.0
            if self.geometry.cfg.optimize_layout:
                means3D[..., 0] = means3D[..., 0] + torch.repeat_interleave(self.geometry.get_layout_depths, self.geometry.mark_instances)
            # in a true iteration only sample two instances.
            if selected_instance_nums[0] == 0:
                pc1 = means3D[: self.mark_accum[0]]
            else:
                i = selected_instance_nums[0]
                pc1 = means3D[self.mark_accum[i - 1]: self.mark_accum[i]]
                
            if selected_instance_nums[1] == 0:
                pc2 = means3D[: self.mark_accum[0]]
            else:
                i = selected_instance_nums[1]
                pc2 = means3D[self.mark_accum[i - 1]: self.mark_accum[i]]
                
            mean_pc2 = torch.mean(pc2, dim=0)
            # pc1 - pc2_m - sp2 + 0.5sp1 , moderate tolerance.
            
            dist_inter = torch.norm(pc1 - mean_pc2, dim=-1, keepdim=True) - \
                         self.disparity_lst[selected_instance_nums[1]] # + \
                         # 0.5 * self.disparity_lst[selected_instance_nums[0]] \
            
            hinge_distances = F.relu(-dist_inter)
            # annealing strategy.
            loss_collision = torch.sum(hinge_distances) * self.cfg.loss['lambda_collision'] * (1 - self.global_step / self.trainer.max_steps)
            self.log(f"train/loss_collision", loss_collision)
            loss_layout += loss_collision
            
        if loss_layout > 0.0:  
            loss_layout.backward(retain_graph=True)
        # =========== Layout refinement losses =========== #

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        loss_sds.backward(retain_graph=True)
        iteration = self.global_step
        self.geometry.update_states(
            iteration,
            visibility_filter,
            radii,
            viewspace_point_tensor,
        )
        if loss > 0:
            loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

        return {"loss": loss_sds}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        # import pdb; pdb.set_trace()
        self.save_image_grid(
            f"it{self.global_step}-{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_pred_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_pred_normal" in out
                else []
            ),
            name="validation_step",
            step=self.global_step,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.global_step}-test/{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_pred_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_pred_normal" in out
                else []
            ),
            name="test_step",
            step=self.global_step,
        )
        if batch["index"][0] == 0:
            save_path = self.get_save_path("point_cloud.ply")
            self.geometry.save_ply(save_path)

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
        print('The optimized 3D layout depths:', self.geometry.get_layout_depths)
