<div align="center">

# Layout-your-3D: Controllable and Precise 3D Generation with 2D Blueprint
[Junwei Zhou](https://github.com/Colezwhy)<sup>1</sup>,[Xueting Li](https://sunshineatnoon.github.io/)<sup>2</sup>,[Lu Qi](http://luqi.info/)<sup>3,4</sup>, [Ming-Hsuan Yang](https://faculty.ucmerced.edu/mhyang/)<sup>5,6</sup>

<sup>1</sup>  Huazhong University of Science and Technology, <sup>2</sup>  NVIDIA, <sup>3</sup>  Wuhan University 

<sup>4</sup> Insta360 Research, <sup>5</sup> UC Merced, <sup>6</sup> Yonsei University

Official implementation for paper 'Layout-Your-3D: Controllable and Precise 3D Generation with 2D Blueprint'.

</div>

## Update
- release ver 0.1: there are still some parts missing or not finished, please stay tuned!
- Layout-Your-3D is accepted by *ICLR 2025*.

## Overview
<div align="center">
<img src="./assets/teaser.png" width="75%" alt="Teaser" align="center">    
</div>

<br>
Given a text prompt describing multiple objects and their 2D spatial relationships, our method generates a 3D scene depicting these objects naturally interacting with one another. Also, Layout-Your-3D supports instance-wise attributes editing and customization, which opens up new possibilities for compositional 3D generation.  

<div align="center">
<div style="width: 80%; margin: 0 auto;">
<table class="center" style="width: 75%">
    <tr style="line-height: 2">
      <td style="width: 30%; border: none; text-align: center">In the background, fireworks bursting. Infront of it, there is a teddy bear dancing slowly within a small area.</td>
      <td style="width: 28%; border: none; text-align: center">An anime girl walking forward
      in a cherry blossom scene.</td>
    </tr>
    <tr style="line-height: 2">
      <td style="width: 30%; border: none"><img src="./assets/squirrel.gif" width="250"></td>
      <td  style="width: 28%; border: none"><img src="./assets/crocodile.gif" width="250"></td>
    </tr>
 </table>
 </div>
</div>

***Check out our [website](https://colezwhy.github.io/layoutyour3d/) and [paper](https://arxiv.org/abs/2410.15391) for more results!***

## Method
<div align="center">
<img src="./assets/pipeline.png" width="80%" alt="Pipeline" align="center">
 </div>

## Installation
We need several repo environments to start our Layout-your-3D codebase.

```bash
## create environment
conda create -n layout3d python==3.9

## ------- LGM ------- ##
cd LGM
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install -U xformers --index-url https://download.pytorch.org/whl/cu118

# a modified gaussian splatting
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization

# for mesh extraction
pip install git+https://github.com/NVlabs/nvdiffrast

# other dependencies
pip install -r requirements.txt
## ------- LGM ------- ##

cd ..

## ------- threestudio ------- ##
cd threestudio
pip install ninja
pip install -r requirements.txt
## ------- threestudio ------- ##
```

## Quick start
We recommend using layout conditioned t2i generation methods like: [MIGC](https://github.com/limuloo/MIGC) and so on to generate the reference images. Here we provide a simple example with MIGC for compositional image generation.

```bash
# First follow MIGC's installation to download the models and checkpoints

# please check the script itself and the notations for more operations, edit the desired layout and prompts from line 29
# ----------- comp img gen ------------ #
cd MIGC 
python ../MIGC/inference_single_image.py

# edit from line 146
# ----------- segment & process ------------ #
python instance_generation.py
# then we use Geowizard for simple depth calculation, we omit this part for simplicity.
```
Also we provide our Comp20 validation set in this link [Comp20](https://drive.google.com/file/d/14dQ1l5qqI7Z-JK-DHG553uyIXDshYaID/view?usp=drive_link). We incorporated all the computed depths and processed images.

After obtaining the prerequisites, we provide the detailed scripts for **each** step for our generation.

```bash
cd ../LGM
# Please follow LGM to download the pretrained models and checkpoints.

# ----------- coarse instance gen ----------- #
python instance_3d.py big --resume pretrained/model_fp16_fixrot.safetensors --workspace ./3d_instance_storage --test_path squirrel

cd ../threestudio
# ----------- refine each instance ----------- #
python launch.py --config custom/threestudio-3dgs/configs/gaussian_config/baseline_refine_ins_short.yaml  --train --gpu 1 system.prompt_processor.prompt="A squirrel" system.geometry.geometry_convert_from="../LGM/3d_instance_storage/squirrel/instance1.ply"
# repeat for each instance...

# or we can optimize the instances with longer refinement strategies.
python launch.py --config custom/threestudio-3dgs/configs/gaussian_config/sd_refinement.yaml  --train --gpu 1 system.prompt_processor.prompt="A squirrel" system.geometry.geometry_convert_from="../LGM/3d_instance_storage/squirrel/instance1.ply"

# refine the layouts, it will print the refined layouts for all three dimension, copy and paste them when fitting in.
python launch.py --config custom/threestudio-3dgs/configs/baseline_layout.yaml  --train --gpu 1 system.prompt_processor.prompt="masterpiece, best quality, A squirrel standing on a box" system.geometry.geometry_convert_from="../LGM/3d_instance_storage/pigeon" 

# batch rotation rectification, refine the poses
python rotation_rectification_batch.py big --resume pretrained/model_fp16_fixrot.safetensors

# after obtaining the layouts, simply render the results with fitting in
python custom/custom3D/fitin_layout.py
python launch.py --config custom/threestudio-3dgs/configs/baseline_eval.yaml  --train --gpu 1 system.prompt_processor.prompt="squirrel" system.geometry.geometry_convert_from="../LGM/example_fin/squirrel/composed.ply"
```

## Applications
For other applications:
- Instance customization: we can simply change the instances' prompts when refining them.
- Object insertion: use the object insertion mode in MIGC can help build this application, generating a base image and then gradually add objects to the reference image.

## Citation
```
@article{zhou2024layout,
  title={Layout-your-3D: Controllable and Precise 3D Generation with 2D Blueprint},
  author={Zhou, Junwei and Li, Xueting and Qi, Lu and Yang, Ming-Hsuan},
  journal={arXiv preprint arXiv:2410.15391},
  year={2024}
}
```

## Acknowledgement
This work is built on [threestudio](https://github.com/threestudio-project/threestudio), [LGM](https://github.com/3DTopia/LGM), and [MIGC](https://github.com/limuloo/MIGC). Thanks all the authors for their contributions!