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

# ============== custom functions ============== #

def resize_for_condition_image(input_image: Image, resolution: int):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img

def centralize(image, mask, border_ratio = 0.2):
    """ recenter an image to leave some empty space at the image border.

    Args:
        image (ndarray): input image, float/uint8 [H, W, 3/4]
        mask (ndarray): alpha mask, bool [H, W]
        border_ratio (float, optional): border ratio, image will be resized to (1 - border_ratio). Defaults to 0.2.

    Returns:
        ndarray: output image, float/uint8 [H, W, 3/4]
    """
    
    return_int = False
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255
        return_int = True
    
    H, W, C = image.shape
    size = max(H, W)
    
    center_pos = int(size / 2)

    # default to white bg if rgb, but use 0 if rgba
    if C == 3:
        result = np.ones((size, size, C), dtype=np.float32)
    else:
        result = np.zeros((size, size, C), dtype=np.float32)
            
    coords = np.nonzero(mask)
    x_min, x_max = coords[0].min(), coords[0].max()
    y_min, y_max = coords[1].min(), coords[1].max()
    h = x_max - x_min
    w = y_max - y_min

    
    desired_size = int(size * (1 - border_ratio))
    scale = desired_size / max(h, w)
    h2 = int(h * scale)
    w2 = int(w * scale)
    x2_min = center_pos - h2 // 2
    x2_max = x2_min + h2
    y2_min = center_pos - w2 // 2
    y2_max = y2_min + w2
    result[x2_min:x2_max, y2_min:y2_max] = cv2.resize(image[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_AREA)

    if return_int:
        result = (result * 255).astype(np.uint8)

    return result

def dilate_and_smooth_mask_in_region(mask, region, size=5, sigma=1):
    # dilation kernel
    kernel = np.ones((size, size), np.uint8)
    
    # define region
    x1, y1, x2, y2 = region
    mask_region = mask[y1:y2, x1:x2]
    print(np.sum(mask_region))
    
    # do dilation
    dilated_mask = cv2.dilate(mask_region, kernel, iterations=1)
    print(np.sum(dilated_mask))
    
    # gaussian kernel
    smoothed_region = cv2.GaussianBlur(dilated_mask.astype(np.float32), (5, 5), 0)
    _, smoothed_mask_binary = cv2.threshold(smoothed_region, 0.5, 1, cv2.THRESH_BINARY)
    print(np.sum(smoothed_mask_binary))
    
    # # into binary
    # binary_region = smoothed_region > 0.5
    
    # # put back to the original mask
    result_mask = mask.copy()
    result_mask[y1:y2, x1:x2] = smoothed_mask_binary.astype(np.uint8)
    print(np.sum(result_mask))
    
    return result_mask

def smooth_mask_in_region(mask, region, kernel_size=7):
    # defining regions
    x1, y1, x2, y2 = region
    mask_region = mask[y1:y2, x1:x2]
    
    # smoothing edge
    smoothed_region = cv2.GaussianBlur(mask_region, (kernel_size, kernel_size), cv2.BORDER_DEFAULT)
    
    # put it back
    result_mask = mask.copy()
    result_mask[y1:y2, x1:x2] = smoothed_region
    
    return result_mask

if __name__ == '__main__':
    work_space = './layout2image'
    
    migc_ckpt_path = 'pretrained_weights/MIGC_SD14.ckpt'
    assert os.path.isfile(migc_ckpt_path), "Please download the ckpt of migc and put it in the pretrained_weighrs/ folder!"
    

    sd1x_path = '/sdb/zdw/weights/stable-diffusion-v1-4' if os.path.isdir('/sdb/zdw/weights/stable-diffusion-v1-4') else "CompVis/stable-diffusion-v1-4"
    # MIGC is a plug-and-play controller.
    # You can go to https://civitai.com/search/models?baseModel=SD%201.4&baseModel=SD%201.5&sortBy=models_v5 find a base model with better generation ability to achieve better creations.
    device = "cuda"
    # Construct MIGC pipeline
    pipe = StableDiffusionMIGCPipeline.from_pretrained(
        sd1x_path)
    pipe.attention_store = AttentionStore()
    from migc.migc_utils import load_migc
    load_migc(pipe.unet , pipe.attention_store,
            migc_ckpt_path, attn_processor=MIGCProcessor)
    pipe = pipe.to(device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    # ================================================= #
    # --------- edit prompt & instances heree --------- #
    prompt_final = [['masterpiece, best quality, front view, A blue toy robot and a red toy robot playing basketball', 'A blue toy robot', 'A red toy robot', 'a basketball']]
    bboxes = [[[0.078, 0.125, 0.46875, 0.8125], [0.5, 0.09375, 0.953125, 0.8125], [0.39, 0.328, 0.59375, 0.53125]]]
    asset = 'play_basketball'
    # --------- edit prompt & instances heree --------- #
    # ================================================= #
    
    Path(f'./layout2image/{asset}').mkdir(parents=True, exist_ok=True)
    
    negative_prompt = 'worst quality, low quality, bad anatomy, watermark, text, blurry'
    seed = 42
    seed_everything(seed)
    image = pipe(prompt_final, bboxes, num_inference_steps=100, guidance_scale=7.5, 
                    MIGCsteps=25, aug_phase_with_and=False, negative_prompt=negative_prompt).images[0]
    image.save(f'./layout2image/{asset}/output.png')
    image = pipe.draw_box_desc(image, bboxes[0], prompt_final[0][1:])
    image.save(f'./layout2image/{asset}/anno_output.png')
    
    # segment output layout-guided t2i results.

    sam_checkpoint = "./sam_weights/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    bg_remover = rembg.new_session()
    input_image = kiui.read_image(f'./layout2image/{asset}/output.png', mode='uint8') # read image in kiui.
    image = cv2.imread(f'./layout2image/{asset}/output.png') # example image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_box = np.array(bboxes[0]) # into numpy bboxes.
    predictor.set_image(image) # set image
    
    # SD inpainting pipeline
    pipe_inpainting = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting", #   "runwayml/stable-diffusion-inpainting"
    torch_dtype=torch.float32,
    ).to(device)
    
    controlnet = ControlNetModel.from_pretrained('lllyasviel/control_v11f1e_sd15_tile', 
                                             torch_dtype=torch.float16)
    pipe_enhance = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
                                            custom_pipeline="stable_diffusion_controlnet_img2img",
                                            controlnet=controlnet,
                                            torch_dtype=torch.float16).to('cuda:0')
    pipe_enhance.enable_xformers_memory_efficient_attention()   
    
    centers = np.stack(((input_box[:, 2] + input_box[:, 0]) / 2, (input_box[:, 1] + input_box[:, 3]) / 2), axis=1)
    for i, instance_box in enumerate(input_box):
        instance_box = instance_box * 512 # input resolution transform the BOX format.
        # SAM segmenting instances.
        center_exclude = np.delete(centers, i, axis=0) * 512
        point_labels = np.zeros(center_exclude.shape[0]).astype(np.uint8)
        mask, _, _ = predictor.predict(
            point_coords=center_exclude,
            point_labels=point_labels,
            box=instance_box[None, :],
            multimask_output=False,
        )
        mask_save = (np.expand_dims(mask[0], 2).repeat(3, axis = 2) * 255).astype(np.uint8)
        mask_save = Image.fromarray(mask_save)
        mask_save.save(os.path.join(work_space, f'{asset}/instance_mask_{i}.png'))
        
        seg = input_image * np.expand_dims(mask[0], 2).repeat(3, axis = 2)
        carved_image = rembg.remove(seg, session=bg_remover) # [H, W, 4]
        inpaint_img = carved_image.astype(np.float32) / 255
        inpaint_img = inpaint_img[..., :3] * inpaint_img[..., 3:4] + (1 - inpaint_img[..., 3:4])
        size = (512, 512, 3)
        noise = np.random.rand(*size)
        inpaint_img_noised = inpaint_img.copy()
        inpaint_img_noised[~(mask[0])] = noise[~(mask[0])]
        
        mask_new = (carved_image[..., -1] > 0)
        kiui.write_image(os.path.join(work_space, f'{asset}/preview_{i}.jpg'), inpaint_img)
        kiui.write_image(os.path.join(work_space, f'{asset}/noised_preview_{i}.jpg'), inpaint_img_noised)
        
        seg_out = recenter(carved_image, mask_new, border_ratio=0.2)
        seg_out = seg_out.astype(np.float32) / 255
        if seg_out.shape[-1] == 4:
                seg_out = seg_out[..., :3] * seg_out[..., 3:4] + (1 - seg_out[..., 3:4])
        kiui.write_image(os.path.join(work_space, f'{asset}/instance_{i}.jpg'), seg_out)
        
        # implementation for ComboVerse inpainting
        
        deviation_scale = int(0)  # 512 / 40
        x1, y1, x2, y2 = instance_box
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        x1, y1, x2, y2 = max(0, x1 - deviation_scale), max(0, y1 - deviation_scale), min(511, x2 + deviation_scale), min(511, y2 + deviation_scale)

        # implementation for inpainting mask.
        mask_inpaint = np.full((512, 512), 0, dtype=np.uint8)
        mask_inpaint[y1:y2, x1:x2] = 1 # set editable regions as a bounding box
        mask_inpaint[mask[0]] = 0
        
        # visualization of the generated mask input. 
        mask_inpaint_vis = np.expand_dims(mask_inpaint, 2).repeat(3, axis = 2) * 255
        msk = Image.fromarray(mask_inpaint_vis)
        msk.save(os.path.join(work_space, f'{asset}/inpaint_mask_{i}.png'))

        prompt = 'a complete 3D model, realistic' + prompt_final[0][i+1]
        
        inpaint_img = kiui.read_image(os.path.join(work_space, f'{asset}/noised_preview_{i}.jpg'), mode='pil') # noised_
        mask_inpaint = kiui.read_image(os.path.join(work_space, f'{asset}/inpaint_mask_{i}.png'), mode='pil')
        
        inpainted_seg = pipe_inpainting(prompt=prompt, image=inpaint_img, mask_image=mask_inpaint, num_inference_steps=100).images[0]
        inpainted_seg.save(f"./layout2image/{asset}/inpainted_instance_{i}.png")
        # rmbg
        inpainted_seg_ = kiui.read_image(f"./layout2image/{asset}/inpainted_instance_{i}.png", mode='uint8')
        carved_image = rembg.remove(inpainted_seg_, session=bg_remover) # [H, W, 4]
        mask_ = (carved_image[..., -1] > 0)
        seg_out1 = recenter(carved_image, mask_, border_ratio=0.2)
        seg_out1 = seg_out1.astype(np.float32) / 255
        seg_out1 = seg_out1[..., :3] * seg_out1[..., 3:4] + (1 - seg_out1[..., 3:4])
        kiui.write_image(os.path.join(work_space, f'{asset}/instance_fin_{i}.jpg'), seg_out1)
        
        # Do enhancement..
        source_image = load_image(os.path.join(work_space, f'{asset}/instance_fin_{i}.jpg'))
        condition_image = resize_for_condition_image(source_image, 1024)
        image = pipe_enhance(prompt=prompt, 
                negative_prompt="blur, lowres, bad anatomy, bad hands, cropped, worst quality", 
                image=condition_image, 
                controlnet_conditioning_image=condition_image, 
                width=condition_image.size[0],
                height=condition_image.size[1],
                strength=1.0,
                generator=torch.manual_seed(0),
                num_inference_steps=32,
                ).images[0]
        Path(os.path.join(work_space, f'{asset}/instance{i}')).mkdir(parents=True, exist_ok=True)
        image.save(os.path.join(work_space, f'{asset}/instance{i}/tiled_inpainted_instance_{i}.png'))
        # Reload and then do centralization.
        cur_img = kiui.read_image(os.path.join(work_space, f'{asset}/instance{i}/tiled_inpainted_instance_{i}.png'), mode = 'uint8')
        carved_image = rembg.remove(cur_img, session=bg_remover) # [H, W, 4]
        mask_new = (carved_image[..., -1] > 0)
        centered_img = centralize(carved_image, mask_new, border_ratio=0.2)
        centered_img = centered_img.astype(np.float32) / 255.0
        if centered_img.shape[-1] == 4:
            centered_img = centered_img[..., :3] * centered_img[..., 3:4] + (1 - centered_img[..., 3:4])
        Path(os.path.join(work_space, f'{asset}/instance{i}')).mkdir(parents = True, exist_ok = True)
        kiui.write_image(os.path.join(work_space, f'{asset}/instance{i}/centralized_tiled.jpg'), centered_img)
    
