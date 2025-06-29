import yaml
from diffusers import EulerDiscreteScheduler
from migc.migc_utils import seed_everything
from migc.migc_pipeline import StableDiffusionMIGCPipeline, MIGCProcessor, AttentionStore
import os
import kiui
from pathlib import Path

if __name__ == '__main__':
    partial = './layout2img'
    migc_ckpt_path = '../MIGC/pretrained_weights/MIGC_SD14.ckpt'
    assert os.path.isfile(migc_ckpt_path), "Please download the ckpt of migc and put it in the pretrained_weighrs/ folder!"


    sd1x_path = '/sdb/zdw/weights/stable-diffusion-v1-4' if os.path.isdir('/sdb/zdw/weights/stable-diffusion-v1-4') else "CompVis/stable-diffusion-v1-4"
    # MIGC is a plug-and-play controller.
    # You can go to https://civitai.com/search/models?baseModel=SD%201.4&baseModel=SD%201.5&sortBy=models_v5 find a base model with better generation ability to achieve better creations.
    
    # Construct MIGC pipeline
    pipe = StableDiffusionMIGCPipeline.from_pretrained(
        sd1x_path)
    pipe.attention_store = AttentionStore()
    from migc.migc_utils import load_migc
    load_migc(pipe.unet , pipe.attention_store,
            migc_ckpt_path, attn_processor=MIGCProcessor)
    pipe = pipe.to("cuda")
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    
    # ================================================= #
    # --------- edit prompt & instances heree --------- #
    asset = 'sofa'
    work_pth = Path(os.path.join(partial, asset)).mkdir(parents=True, exist_ok=True)
    prompt_final = [['', 'A brown sofa', 'A teddy bear', 'A cushion', 'A basketball']]
    bboxes = [[[0.0625, 0.5, 0.9375, 0.84375], [0.390625, 0.234375, 0.640625, 0.671875], [0.140625, 0.4375, 0.328125, 0.703125], [0.703125, 0.5, 0.890625, 0.6875]]]
    # --------- edit prompt & instances heree --------- #
    # ================================================= #
    
    negative_prompt = 'worst quality, low quality, bad anatomy, watermark, text, blurry'
    seed = 0
    seed_everything(seed)
    image = pipe(prompt_final, bboxes, num_inference_steps=50, guidance_scale=7.5, 
                    MIGCsteps=25, NaiveFuserSteps=25, aug_phase_with_and=False, negative_prompt=negative_prompt).images[0]
    
    image.save(os.path.join(work_pth, 'output.png'))
