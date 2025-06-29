import yaml
from diffusers import EulerDiscreteScheduler, ControlNetModel, DiffusionPipeline
from migc.migc_utils import seed_everything
from migc.migc_pipeline import StableDiffusionMIGCPipeline, MIGCProcessor, AttentionStore
from diffusers.utils import load_image
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor
import kiui
from kiui.op import recenter
import rembg
from diffusers import StableDiffusionInpaintPipeline
import PIL.Image as Image
from pathlib import Path
from skimage import io, morphology, filters

bg_remover = rembg.new_session()
path = '../MIGC/4D_trial_images/instance_whitebg.png'
input_image = kiui.read_image(path, mode='uint8') # read into uint8 format.(255 based)
# bg removal
carved_image = rembg.remove(input_image, session=bg_remover) # [H, W, 4]
mask = carved_image[..., -1] > 0

# recenter
saving_path = '../MIGC/layout2img'
image = recenter(carved_image, mask, border_ratio=0.2)
image = image.astype(np.float32) / 255
if image.shape[-1] == 4:
    image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])
kiui.write_image(os.path.join(saving_path, 'recenter.jpg'), image)