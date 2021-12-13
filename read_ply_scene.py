# alpha version to render scene mesh using ground truth camera parameters
# current results are weird
import os
import torch
import numpy as np
from tqdm.notebook import tqdm
import imageio
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
import cv2

# io utils
from pytorch3d.io import load_obj, load_ply

# datastructures
from pytorch3d.structures import Meshes

# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate

# rendering components
from pytorch3d.renderer import (
    PerspectiveCameras,
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,
)

from plyfile import PlyData

def read_ply_rgb(filename):
    """ read XYZ RGB normals point cloud from filename PLY file """
    assert(os.path.isfile(filename))
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,2] = plydata['vertex'].data['y']
        vertices[:,1] = plydata['vertex'].data['z']
        vertices[:,3] = plydata['vertex'].data['red']
        vertices[:,4] = plydata['vertex'].data['green']
        vertices[:,5] = plydata['vertex'].data['blue']

        # # compute normals
        # xyz = np.array([[x, y, z] for x, y, z, _, _, _, _ in plydata["vertex"].data])
        faces = np.array([f[0] for f in plydata["face"].data])
        # nxnynz = compute_normal(xyz, face)
        # vertices[:,6:] = nxnynz
    return vertices, faces

# Set the cuda device 
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# Load the obj and ignore the textures and materials.
ply_path = "/home/siyich/ScanNet/scans/scene0000_00/scene0000_00_vh_clean_2.labels.ply"
pose_path = "/home/siyich/ScanNet/scans/scene0000_00/frames/pose/0.txt"
intrin_path = "/home/siyich/ScanNet/scans/scene0000_00/frames/intrinsic/intrinsic_color.txt"


verts1, faces1 = load_ply(ply_path)
verts_rgb1 = torch.ones_like(verts1)[None]  # (1, V, 3)
textures1 = TexturesVertex(verts_features=verts_rgb1.to(device))


texture_vertices, faces = read_ply_rgb(ply_path)
verts = torch.from_numpy(texture_vertices[:,:3])

faces = torch.from_numpy(faces)
verts_rgb = torch.from_numpy(texture_vertices[:,3:6]/255) # Need to convert to range (0,1)
print(torch.max(verts_rgb))
verts_rgb = verts_rgb[None]
textures = TexturesVertex(verts_features=verts_rgb.to(device))

# print(faces1 == faces)
# print(verts1 == verts)


# Create a Meshes object for the teapot. Here we have only one mesh in the batch.
scene_mesh = Meshes(
    verts=[verts.to(device)],   
    faces=[faces.to(device)], 
    textures=textures
)

intrinsic = np.loadtxt(intrin_path)
foc = (intrinsic[0,0],intrinsic[1,1])
c = (intrinsic[0,2],intrinsic[1,2])
w = 640 * 2
h = 480 * 2
image_size = ((h, w),)

fcl_ndc = (foc[0] * 2 / (h-1), foc[1] * 2 / (h-1))
px_ndc = ((w-1)/2 - w/2) * 2 / (h-1)
py_ndc = ((h-1)/2 - h/2) * 2 / (h-1)
prp_ndc = (px_ndc, py_ndc)


# Initialize a perspective camera.
cameras = FoVPerspectiveCameras(device=device)
# cameras = PerspectiveCameras(focal_length = (foc,), principal_point = (c,), device=device)
# cameras = PerspectiveCameras(principal_point = (c,), in_ndc = False, image_size=image_size, device=device)
# cameras = PerspectiveCameras(focal_length=(fcl_ndc,), principal_point=(prp_ndc,), device=device)

blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

raster_settings = RasterizationSettings(
    image_size=(h,w), 
    blur_radius=0.0, 
    faces_per_pixel=1, 
    perspective_correct=False,
)
# We can add a point light in front of the object. 
lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
phong_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=HardPhongShader(device=device, cameras=cameras, lights=lights)
)

# Select the viewpoint using spherical angles  
distance = 10   # distance from camera to the object
elevation = 0.0   # angle of elevation in degrees
azimuth = 0.0  # No rotation so the camera is positioned on the +Z axis.

# Axis alignment
# 0.945519 0.325568 0.000000 -5.384390 
# -0.325568 0.945519 0.000000 -2.871780 
# 0.000000 0.000000 1.000000 -0.064350 
# 0.000000 0.000000 0.000000 1.000000

# Get the position of the camera based on the spherical angles
R, T = look_at_view_transform(distance, elevation, azimuth, device=device)
print(R,T)

# extrinsic = np.loadtxt(pose_path)
# R = torch.from_numpy(extrinsic[None,:3,:3]).to(device)
# T = torch.from_numpy(extrinsic[None,:3,3]).to(device)
# print(R,T)


image_ref = phong_renderer(meshes_world=scene_mesh, R=R, T=T)
# image_ref = phong_renderer(meshes_world=scene_mesh)
image_ref = image_ref.cpu().numpy()
image_ref_save = image_ref[0][..., :3]
image_path = 'scene.png'
cv2.imwrite(image_path, image_ref_save * 255)




verts, faces_idx, _ = load_obj("./data/teapot.obj")
faces = faces_idx.verts_idx

# Initialize each vertex to be white in color.
verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
textures = TexturesVertex(verts_features=verts_rgb.to(device))

# Create a Meshes object for the teapot. Here we have only one mesh in the batch.
teapot_mesh = Meshes(
    verts=[verts.to(device)],   
    faces=[faces.to(device)], 
    textures=textures
)

image_ref = phong_renderer(meshes_world=teapot_mesh, R=R, T=T)
# image_ref = phong_renderer(meshes_world=scene_mesh)
image_ref = image_ref.cpu().numpy()
image_ref_save = image_ref[0][..., :3]
image_path = 'teapot.png'
cv2.imwrite(image_path, image_ref_save * 255)