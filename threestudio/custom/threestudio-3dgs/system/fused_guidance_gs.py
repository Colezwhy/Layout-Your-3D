import os
import torch
from dataclasses import dataclass, field
from threestudio.utils.loss import tv_loss
import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.misc import cleanup, get_device
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *


@threestudio.register("fused-gaussian-system")
class FusedGSSystem(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        visualize_samples: bool = False
        latent_steps: int = 1000
        
        guidance_2d_type: str = ""
        guidance_2d: dict = field(default_factory=dict)
        
        prompt_2d_processor_type: str = ""
        prompt_2d_processor: dict = field(default_factory=dict)

    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()
        self.has_mv_guidanece = (self.cfg.guidance_type != "none") and hasattr(
            self.cfg.loss, "lambda_sds"
        )  # and (self.cfg.loss.lambda_nd > 0)
        self.has_sd_guidanece = (self.cfg.guidance_2d_type != "none") and hasattr(
            self.cfg.loss, "lambda_2d_sds"
        )  # and (self.cfg.loss.lambda_rgb_sd > 0)
        threestudio.info(
            f"================has mv guidance:{self.has_mv_guidanece}, has sd guidance:{self.has_sd_guidanece}================="
        )

        if self.has_sd_guidanece:
            self.guidance_SD = threestudio.find(self.cfg.guidance_2d_type)(self.cfg.guidance_2d)
            self.prompt_processor_SD = threestudio.find(self.cfg.prompt_2d_processor_type)(
                self.cfg.prompt_2d_processor
            )
            self.prompt_utils_SD = self.prompt_processor_SD()

        if self.has_mv_guidanece:
            self.guidance_MV = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
            self.prompt_processor_MV = threestudio.find(self.cfg.prompt_processor_type)(
                self.cfg.prompt_processor
            )
            self.prompt_utils_MV = self.prompt_processor_MV()

    # def on_load_checkpoint(self, checkpoint):
    #     for k in list(checkpoint["state_dict"].keys()):
    #         if k.startswith("guidance."):
    #             return
    #         if k.startswith("guidance_2d."):
    #             return
    #     if self.has_rgb_sd_guidanece:
    #         if hasattr(self.guidance, "state_dict"):
    #             guidance_state_dict = {
    #                 "guidance." + k: v for (k, v) in self.guidance.state_dict().items()
    #             }
    #             checkpoint["state_dict"] = {
    #                 **checkpoint["state_dict"],
    #                 **guidance_state_dict,
    #             }

    #     if self.has_nd_guidanece:
    #         guidance_nd_state_dict = {
    #             "guidance_2d." + k: v
    #             for (k, v) in self.nd_guidance.state_dict().items()
    #         }
    #         checkpoint["state_dict"] = {
    #             **checkpoint["state_dict"],
    #             **guidance_nd_state_dict,
    #         }

    #     return

    # def on_save_checkpoint(self, checkpoint):
    #     for k in list(checkpoint["state_dict"].keys()):
    #         if k.startswith("guidance."):
    #             checkpoint["state_dict"].pop(k)
    #         if k.startswith("guidance_2d."):
    #             checkpoint["state_dict"].pop(k)
    #     return
    
    def on_load_checkpoint(self, checkpoint):
        for k in list(checkpoint['state_dict'].keys()):
            if k.startswith("guidance."):
                return
        guidance_state_dict = {"guidance."+k : v for (k,v) in self.guidance.state_dict().items()}
        checkpoint['state_dict'] = {**checkpoint['state_dict'], **guidance_state_dict}
        return

    def on_save_checkpoint(self, checkpoint):
        for k in list(checkpoint['state_dict'].keys()):
            if k.startswith("guidance."):
                checkpoint['state_dict'].pop(k)
        return

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        self.geometry.update_learning_rate(self.global_step)
        outputs = self.renderer.batch_forward(batch)
        return outputs

    def on_fit_start(self) -> None:
        super().on_fit_start()

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        out = self(batch)
        visibility_filter = out["visibility_filter"]
        radii = out["radii"]
        guidance_inp = out["comp_rgb"]
        viewspace_point_tensor = out["viewspace_points"]

        # Enable both 2d and 3d guidance.
        self.has_mv_guidanece = (
            (self.cfg.guidance_type != "none")
            and hasattr(self.cfg.loss, "lambda_sds")
            and (self.C(self.cfg.loss.lambda_sds) > 0)
        )
        self.has_sd_guidanece = (
            (self.cfg.guidance_2d_type != "none")
            and hasattr(self.cfg.loss, "lambda_2d_sds")
            and (self.C(self.cfg.loss.lambda_2d_sds) > 0)
        )

        if self.has_mv_guidanece:   
            mv_guidance_out = self.guidance_MV(
                guidance_inp,
                self.prompt_utils_MV,
                **batch,
                rgb_as_latents=False,
            )

        if self.has_sd_guidanece:
            sd_guidance_out = self.guidance_SD(
                guidance_inp,
                self.prompt_utils_SD,
                **batch,
                rgb_as_latents=False,
            )
            
        loss = 0.0
        
        self.log(
            "gauss_num",
            int(self.geometry.get_xyz.shape[0]),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        
        # 3D guidance
        if self.has_mv_guidanece:
            for name, value in mv_guidance_out.items():
                if 'norm' in name:
                    name1 = name.replace("norm", "norm3d")
                    self.log(f"train/{name1}", value)
                if name.startswith("loss_"):
                    loss_3d = value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])
                    self.log(f"train/loss_3d", loss_3d)
                    loss += loss_3d

        # 2D guidance
        if self.has_sd_guidanece:
            for name, value in sd_guidance_out.items():
                self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                    loss_2d = value * self.C(self.cfg.loss[name.replace("loss_", "lambda_2d_")])
                    self.log(f"train/loss_2d", loss_2d)
                    loss += loss_2d

        
        # ADDED FOR SUPPORTING GAUSSIAN SPLATTING.
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
        
        # normal smoothness loss.
        if self.C(self.cfg.loss.lambda_normal_smooth_loss) > 0:
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

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))
        
        loss.backward(retain_graph=True)
        iteration = self.global_step
        self.geometry.update_states(
            iteration,
            visibility_filter,
            radii,
            viewspace_point_tensor,
        )
            
        opt.step()
        opt.zero_grad(set_to_none=True)
        
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step:06d}-{batch['index'][0]:04d}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
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
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step:06d}-test/{batch['index'][0]:04d}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
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
            step=self.true_global_step,
        )
        if batch["index"][0] == 0:
            save_path = self.get_save_path("point_cloud.ply")
            self.geometry.save_ply(save_path)

    def on_test_epoch_end(self):
        self.save_img_sequence(
                f"it{self.true_global_step:06d}-test",
                f"it{self.true_global_step:06d}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )