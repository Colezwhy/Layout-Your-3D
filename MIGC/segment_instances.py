import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor
import kiui
from kiui.op import recenter
from PIL import Image
import os

sam_checkpoint = "./sam_weights/sam_vit_h_4b8939.pth"
model_type = "vit_h"

work_space = '../MIGC/layout2image'
asset = 'shoe'

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

image = cv2.imread(f'../MIGC/layout2image/{asset}/output.png') # example image
input_image = kiui.read_image(f'../MIGC/layout2image/{asset}/output.png', mode='uint8')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

input_boxes = [[0.203125, 0.28125, 0.828125, 0.8125], [0.25, 0.078125, 0.53125, 0.28125], [0.59375, 0.109375, 0.765625, 0.296875] ]
input_box = np.array(input_boxes) * 512

blank =  kiui.read_image('../stabilityai/generative-models/bg.png', mode='uint8')
blank = np.full_like(blank, 0).astype(np.uint8)

predictor.set_image(image)

for i, instance_box in enumerate(input_box):
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=instance_box,
        multimask_output=False,
    )
    mask_save = (np.expand_dims(masks[0], 2).repeat(3, axis = 2) * 255).astype(np.uint8)
    mask_save = Image.fromarray(mask_save)
    mask_save.save(os.path.join(work_space, f'{asset}/instance_mask_{i}.png'))
    mask_bool = np.expand_dims(masks[0], 2).repeat(3, axis = 2)
    seg = input_image * mask_bool
    blank = blank * (~mask_bool) + seg
    
kiui.write_image(os.path.join(work_space, f'{asset}/foreground_composed.jpg'), blank)