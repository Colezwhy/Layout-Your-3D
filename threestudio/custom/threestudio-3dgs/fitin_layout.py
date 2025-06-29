from plyfile import PlyData, PlyElement
import numpy as np
import torch
import os
from pathlib import Path
import kiui
import kiui.op
from kiui.op import recenter
from pathlib import Path

'''
for LGM generated gaussians, the attributes of gaussians are as follows:
[0:3]: positions,
[3:4]: opacitys,
[4:7]: scales,
[7:11]: rotations,
[11:14]: rgbs
Coarsely, we can assume that the value of coordinates are in a range[-0.5, 0.5]
'''


#@ TODO: Adjust this fit in part, maybe should take the center of the mask as the layout centers, also, the scales should be considered.

def ratio(box, size = 512, border_ratio = 0.2):
    """ 
    Adjusted from recenter, for getting transformation scaling factor.
    """
    box = box * size
    x_min, x_max = box[0], box[2]
    y_min, y_max = box[1], box[3]
    h = x_max - x_min
    w = y_max - y_min
    desired_size = int(size * (1 - border_ratio))
    scale = desired_size / max(h, w) / 0.8
    h2 = int(h * scale)
    w2 = int(w * scale)
    
    ratio = scale
    # x2_min = (size - h2) // 2
    # x2_max = x2_min + h2
    # y2_min = (size - w2) // 2
    # y2_max = y2_min + w2
    # result[x2_min:x2_max, y2_min:y2_max] = cv2.resize(image[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_AREA)

    return ratio

def transform_gaussians(gaussians, scale_factor, box_coordinates):
    '''
    For instance we only adjust the absolute positions and scaling factors.
    '''
    x, y, z = box_coordinates[-3:]
    gaussians[..., 0:3] = gaussians[..., 0:3] * scale_factor  # Adjusting the xyz positions
    gaussians[..., 4:7] *= scale_factor  # Adjusting the scales
    gaussians[..., 0] += x
    gaussians[..., 1] -= y
    gaussians[..., 2] += z # here according to the defined coordinate system, we must minus the y for instance.

    
        # Adjust the color if necessary (optional)
        # gaussian['color'] = adjust_color(gaussian['color'], scale_factor)
        
    return gaussians

def save_ply(gaussians, path, compatible=True):
        # gaussians: [B, N, 14]
        # compatible: save pre-activated gaussians as in the original paper

        assert gaussians.shape[0] == 1, 'only support batch size 1'

        from plyfile import PlyData, PlyElement
     
        means3D = gaussians[0, :, 0:3].contiguous().float()
        opacity = gaussians[0, :, 3:4].contiguous().float()
        scales = gaussians[0, :, 4:7].contiguous().float()
        rotations = gaussians[0, :, 7:11].contiguous().float()
        shs = gaussians[0, :, 11:].unsqueeze(1).contiguous().float() # [N, 1, 3]

        # =============== Gaussians for instance generation =============== #
        # prune by opacity, finalizing the formulation of gaussians.
        # I.E. the original resolution of the whole space is 256 ** 2 = 65536 in which each pixel corresponds to one gaussian.
        mask = opacity.squeeze(-1) >= 0.005  # Here only retain those gaussians that have opacity larger than 0.005 as spatial representation.
        means3D = means3D[mask] # coordinates
        opacity = opacity[mask] # opacities
        scales = scales[mask]
        rotations = rotations[mask]
        shs = shs[mask]
        # =============== Gaussians for instance generation =============== #   

        # invert activation to make it compatible with the original ply format
        if compatible:
            opacity = kiui.op.inverse_sigmoid(opacity)
            scales = torch.log(scales + 1e-8)
            shs = (shs - 0.5) / 0.28209479177387814

        xyzs = means3D.detach().cpu().numpy()
        f_dc = shs.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy() # Spherical harmonics
        opacities = opacity.detach().cpu().numpy()
        scales = scales.detach().cpu().numpy()
        rotations = rotations.detach().cpu().numpy()

        l = ['x', 'y', 'z']
        # All channels except the 3 DC
        for i in range(f_dc.shape[1]):
            l.append('f_dc_{}'.format(i))
        l.append('opacity')
        for i in range(scales.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rotations.shape[1]):
            l.append('rot_{}'.format(i))

        dtype_full = [(attribute, 'f4') for attribute in l]

        elements = np.empty(xyzs.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyzs, f_dc, opacities, scales, rotations), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex') # writing elements of gaussians, representing a single instance.

        PlyData([el]).write(path)

def load_ply(path, compatible=True):

        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        print("Number of points at loading : ", xyz.shape[0])

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        shs = np.zeros((xyz.shape[0], 3))
        shs[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        shs[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
        shs[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
          
        gaussians = np.concatenate([xyz, opacities, scales, rots, shs], axis=1)
        gaussians = torch.from_numpy(gaussians).float() # cpu

        if compatible:
            gaussians[..., 3:4] = torch.sigmoid(gaussians[..., 3:4])
            gaussians[..., 4:7] = torch.exp(gaussians[..., 4:7])
            gaussians[..., 11:] = 0.28209479177387814 * gaussians[..., 11:] + 0.5

        return gaussians
    
def depth_map_to_3d(depth_map, K, depth_min, depth_max):
    """
    Convert a depth map to 3D coordinates.
    
    Parameters:
    - depth_map: (H, W) array of normalized depth values [0, 1]
    - K: Intrinsic matrix (3x3)
    - depth_min: Minimum depth value
    - depth_max: Maximum depth value
    
    Returns:
    - points_3d: (H, W, 3) array of 3D coordinates
    """
    H, W = depth_map.shape
    i, j = np.indices((H, W))
    normalized_depth = depth_map
    depth = depth_min + normalized_depth * (depth_max - depth_min)
    
    # Intrinsic parameters
    f_x = K[0, 0]
    f_y = K[1, 1]
    c_x = K[0, 2]
    c_y = K[1, 2]
    
    # Calculate 3D coordinates
    x = (j - c_x) / f_x
    y = (i - c_y) / f_y
    X = x * depth
    Y = y * depth
    Z = depth
    
    points_3d = np.stack((X, Y, Z), axis=-1)
    return points_3d

def calculate_intrinsic_matrix(fovy, width, height):
    """
    Calculate the intrinsic matrix K given fovy, width, and height.

    Parameters:
    - fovy: Field of view in the y direction (in radians)
    - width: Image width (in pixels)
    - height: Image height (in pixels)

    Returns:
    - K: Intrinsic matrix (3x3)
    """
    # Calculate focal lengths
    f_y = height / (2 * np.tan(fovy / 2))
    f_x = f_y * (width / height)

    # Principal point (usually the center of the image)
    c_x = width / 2
    c_y = height / 2

    # Intrinsic matrix
    K = np.array([
        [f_x, 0, c_x],
        [0, f_y, c_y],
        [0, 0, 1]
    ])

    return K

folder_path = '../LGM/example_fin/squirrel' # /high_quality/custom1
img_path = '../MIGC/layout2image/squirrel'

msks = os.listdir(img_path)
msks.sort()

depth_map = os.path.join(img_path, 'depth.npy')
depth_map = np.load(depth_map)

files = os.listdir(folder_path)
files.sort()

rotation_matrix_90_left = torch.tensor([
    [0, 0, 1],
    [0, 1, 0],
    [-1, 0, 0]
]).to(torch.float32)

# =============== FOR GAUSSIAN LOADING ============== #
gaussian_pth = []
gaussians = []
for i, file in enumerate(files):
    if 'instance' not in file:
        continue
    if 'ply' not in file:
        continue
    if 'rectified' not in file: # for final results' display
        continue
    gau_pth = os.path.join(folder_path, file)
    gaussian_pth.append(gau_pth)
    gau = load_ply(gau_pth)
    # if '1' in file:
    #     # gau[..., 2] = - gau[..., 2]
    #     # gau[..., 0] = - gau[..., 0]
    #     gau[..., :3] = torch.matmul(gau[..., :3], rotation_matrix_90_left.T)
    #     save_ply(torch.unsqueeze(gau, dim=0), os.path.join(folder_path, 'instance1.ply'))
    gaussians.append(gau)
    

# =============== FOR DEPTH CALCULATION =============== #

fovy = np.radians(49.1)  # 
width = 512
height = 512
K = calculate_intrinsic_matrix(fovy, width, height)
depth_min = np.min(depth_map)
depth_max = np.max(depth_map)
depth_map = depth_map_to_3d(depth_map, K, depth_min = depth_min, depth_max = depth_max)
masks = []
depths=  []
for msk in msks:
    if 'instance' not in msk:
        continue
    if 'mask' not in msk:
        continue
    mask = kiui.read_image(os.path.join(img_path, msk), mode='uint8')
    mask = mask[..., 0].astype(np.bool_)
    masks.append(mask)
    
    mask_depth = depth_map[..., -1] * mask
    mask_mean = 1 - np.sum(mask_depth) / np.sum(mask)
    depths.append(mask_mean)
depths = np.array(depths)
 
# ==================== updated depth layouts ====================== #
depths += np.array([-0.0016,  0.0586]) # fit in the layouts. pigeon [-0.1306,  0.1400,  0.0300] beagle [-0.0016,  0.1986] [0.001, 0.003, 0.1231]
# ==================== updated depth layouts ====================== #
# normalization
m_d = np.sum(depths) / depths.shape[0]
depths = depths - m_d
    


# =============== FOR DEPTH CALCULATION =============== #

# =============== for layout initialization ================ #
layout_2d_boxes = np.array([[0.078125, 0.390625, 0.921875, 0.84375], [0.25, 0.09375, 0.734375,0.390625]]) # 
scaling_factors = []
for i, item in enumerate(layout_2d_boxes):
    scal = ratio(item)
    if i == 0:
        scaling_factors.append(1 / (scal))
    else:
        scaling_factors.append(1 / scal)

box_config = []

for i, item in enumerate(layout_2d_boxes):
    w = item[2] - item[0]
    h = item[3] - item[1]
    l = w
    x = (item[2] + item[0]) / 2 - 0.5
    y = (item[3] + item[1]) / 2 - 0.5
    z = depths[i]
    box_info = [w, h, l, x, y, z]
    box_config.append(box_info) 
    
print(box_config)

instances = []

for i, item in enumerate(gaussians):
    if i >= len(layout_2d_boxes):
        print('skipping')
        continue
    print(item.shape[0])
    instance = transform_gaussians(item, scaling_factors[i], box_config[i])
    instances.append(instance)
    
fitin_gaussians = torch.cat(instances, dim=0)
save_ply(torch.unsqueeze(fitin_gaussians, dim=0), os.path.join(folder_path, 'composed.ply'))
