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


def ratio(box, size = 512, border_ratio = 0.2):
    """ 
    Adjusted from recenter, for getting transformation scaling_factor and then fit into 3D layout.
    """
    box = box * size
    x_min, x_max = box[0], box[2]
    y_min, y_max = box[1], box[3]
    h = x_max - x_min
    w = y_max - y_min
    desired_size = int(size * (1 - border_ratio))
    scale = desired_size / max(h, w)
    # h2 = int(h * scale)
    # w2 = int(w * scale)    
    ratio = scale
    return ratio

def transform_gaussians(gaussians, scale_factor, box_coordinates):
    '''
    For instance we only adjust the absolute positions and scaling factors.
    '''
    x, y, z = box_coordinates[-3:]
    gaussians[..., 0:3] = gaussians[..., 0:3] * scale_factor  # Adjusting the xyz positions with hadamard product
    gaussians[..., 4:7] *= scale_factor  # Adjusting the scales with quarternion hadamard.
    gaussians[..., 0] += x
    gaussians[..., 1] -= y # here according to the defined coordinate system, we must minus the y for instance.
    gaussians[..., 2] += z 
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

def initialize_3d_layouts(layout_boxes, init_depths=None):
    '''
    Initializing 3D layouts from given 2D layouts, here we simply make some trivial assumptions.
    '''
    box_config = []

    for i, item in enumerate(layout_boxes):
        w = item[2] - item[0]
        h = item[3] - item[1]
        l = w
        x = (item[2] + item[0]) / 2 - 0.5
        y = (item[3] + item[1]) / 2 - 0.5
        if init_depths is None:
            z = 0.0
        else:
            z = init_depths[i]
        box_info = [w, h, l, x, y, z]
        box_config.append(box_info)
    return box_config
    

def quaternion_to_rotation_matrix(q):
    # get quaternion transformation matrix.
    w, x, y, z = q
    return torch.tensor([
        [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)]
    ]).to('cuda')


def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R



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