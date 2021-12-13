# render shapenet with/without mesh

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm
import imageio
from skimage import img_as_ubyte
from scipy import ndimage

from pytorch3d.datasets import (
    R2N2,
    ShapeNetCore,
    collate_batched_meshes,
    render_cubified_voxels,
)
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    OpenGLPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    TexturesVertex,
    look_at_view_transform,
    MeshRenderer,
    BlendParams,
    MeshRasterizer,
    SoftSilhouetteShader,
)

from pytorch3d.structures import Meshes
from torch.utils.data import DataLoader

# add path for demo utils functions 
import sys
import os

import matplotlib.pyplot as plt

import cv2
from glob import glob
from PIL import Image, ImageColor

def image_grid(
    images,
    rows=None,
    cols=None,
    fill: bool = True,
    show_axes: bool = False,
    rgb: bool = True,
):
    """
    A util function for plotting a grid of images.
    Args:
        images: (N, H, W, 4) array of RGBA images
        rows: number of rows in the grid
        cols: number of columns in the grid
        fill: boolean indicating if the space between images should be filled
        show_axes: boolean indicating if the axes of the plots should be visible
        rgb: boolean, If True, only RGB channels are plotted.
            If False, only the alpha channel is plotted.
    Returns:
        None
    """
    if (rows is None) != (cols is None):
        raise ValueError("Specify either both rows and cols or neither.")

    if rows is None:
        rows = len(images)
        cols = 1

    gridspec_kw = {"wspace": 0.0, "hspace": 0.0} if fill else {}
    fig, axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw, figsize=(15, 9))
    bleed = 0
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    idx = 0
    for ax, im in zip(axarr.ravel(), images):
        if rgb:
            # only render RGB channels
            # ax.imshow(im[..., :3])
            im0 = im[..., :3] * 255
            im_bgr = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR) # OpenCV is BGR, Pillow is RGB
            cv2.imwrite(f"rendered{idx}.png", im_bgr)
            
        else:
            # only render Alpha channel
            # ax.imshow(im[..., 3])
            cv2.imwrite(f"alpha{idx}.png", im[..., 3] * 255)
            
        if not show_axes:
            ax.set_axis_off()
        idx += 1


class Model(nn.Module):
    def __init__(self, meshes, renderer, image_ref, R, T):
        super().__init__()
        self.meshes = meshes
        self.device = meshes.device
        self.renderer = renderer
        
        # Get the silhouette of the reference RGB image by finding all non-white pixel values. 
        image_ref = torch.from_numpy(image_ref.astype(np.float32))
        image_ref = image_ref[None]
        self.register_buffer('image_ref', image_ref)

        image_help_w = np.ones((480,640))
        image_help_h = np.ones((480,640))
        for i in range(480):
            image_help_w[i,:] = i * image_help_w[i,:]
        for j in range(640):
            image_help_h[:,j] = j * image_help_h[:,j]
        image_help_w = torch.from_numpy(image_help_w)
        image_help_w = image_help_w[None]
        self.register_buffer('image_help_w', image_help_w)
        image_help_h = torch.from_numpy(image_help_h)
        image_help_h = image_help_h[None]
        self.register_buffer('image_help_h', image_help_h)

        # Optimize R and T directly
        self.R = nn.Parameter(R.to(meshes.device)) # (1, 3, 3)
        self.T = nn.Parameter(T.to(meshes.device)) # (1, 3)


    def forward(self):
        
        # Render the image using the updated camera position. Based on the new position of the 
        # camera we calculate the rotation and translation matrices
        
        
        image = self.renderer(meshes_world=self.meshes.clone(), R=self.R, T=self.T)

        # print(image.size(), self.image_ref.size())
        
        # Calculate the silhouette loss
        loss_inter = (torch.sum(image[..., 3] * self.image_ref) / torch.sum(image[..., 3]) - 0.5) ** 2
        loss_center = (torch.mean(image[..., 3] * self.image_help_w) - torch.mean(self.image_ref * self.image_help_w)) ** 2 + (torch.mean(image[..., 3] * self.image_help_h) - torch.mean(self.image_ref * self.image_help_h)) ** 2
        loss = loss_inter + loss_center
        return loss, image


        

# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
    
SHAPENET_PATH = "/home/siyich/siyich/ShapeNetCore.v2"
shapenet_dataset = ShapeNetCore(SHAPENET_PATH, version=2, load_textures=True)

scene_list_path = f'/z/syqian/ScanNet/scans/scene*'
scan_scene_sets = glob(scene_list_path)


save_dir_root = "/data/siyich/am3s"
number_of_data = 100


for ndx in range(number_of_data):

    # *************** Select a background scene image *********************
    scene_root_folder = scan_scene_sets[np.random.randint(len(scan_scene_sets))]
    which_scene = os.path.basename(scene_root_folder)
    print(which_scene)
    frame_id = 0
    background_folder = os.path.join(scene_root_folder, 'frames')
    background_intrinsic = os.path.join(scene_root_folder, 'frames/intrinsic/intrinsic_color.txt')
    background_pose = os.path.join(scene_root_folder, f'frames/pose/{frame_id}.txt')
    background_path = os.path.join(scene_root_folder, f'frames/color/{frame_id}.jpg')
    background_seg = os.path.join(scene_root_folder, f'annotation/segmentation/{frame_id}.png')
    back_im = Image.open(background_path)
    back_seg = cv2.imread(background_seg)
    ground_color = back_seg[450,320,:]
    back_seg[np.where(back_seg[:,:,0] != ground_color[0])] = (0,0,0)

    back_seg_ref = back_seg[:,:,0]
    back_seg_ref[back_seg_ref > 0] = 1.0
    # print(back_seg_ref.shape)
    (width_back, height_back) = (640,480)
    newsize = (width_back, height_back)
    back_im = back_im.resize(newsize)
    # *************** Select a background scene image *********************


    # **************** read a single shape and create mesh *****************
    shapenet_model = shapenet_dataset[np.random.randint(len(shapenet_dataset))]
    print("This model belongs to the category " + shapenet_model["synset_id"] + ".")
    print("This model has model id " + shapenet_model["model_id"] + ".")
    model_verts, model_faces = shapenet_model["verts"], shapenet_model["faces"]
    model_id = shapenet_model["model_id"]

    output_folder = os.path.join(save_dir_root, model_id+"_"+which_scene)
    try:
        os.mkdir(output_folder)
    except:
        pass

    cv2.imwrite(os.path.join(output_folder, "ground.png"), back_seg)
    back_im.save(os.path.join(output_folder, "back.png"))

    model_textures = TexturesVertex(verts_features=torch.ones_like(model_verts, device=device)[None])
    shapenet_model_mesh = Meshes(
        verts=[model_verts.to(device)],   
        faces=[model_faces.to(device)],
        textures=model_textures
    )
    # **************** read a single shape and create mesh *****************



    # Rendering settings.
    distance = np.random.rand() + 1.5   # distance from camera to the object
    elevation = np.random.rand() * 20.0   # angle of elevation in degrees
    azimuth = np.random.rand() * 30.0 - 15.0  # No rotation so the camera is positioned on the +Z axis. 
    R, T = look_at_view_transform(distance, elevation, azimuth)
    T = torch.rand(1,3) - 0.5 + T

    # **************** render with texture *****************
    cameras_visual = FoVPerspectiveCameras(R=R, T=T, device=device)
    raster_settings_visual = RasterizationSettings(image_size=(height_back, width_back), cull_backfaces=True)
    lights = PointLights(
        device=device, 
        location=[[0.0, 5.0, -10.0]], 
        diffuse_color=((0, 0, 0),),
        specular_color=((0, 0, 0),),
    )
    images_by_model_ids = shapenet_dataset.render(
        model_ids=[
            model_id,
        ],
        device=device,
        cameras=cameras_visual,
        raster_settings=raster_settings_visual,
        lights=lights,
    )
    image = images_by_model_ids[0, ..., :3].detach().squeeze().cpu().numpy() * 255
    cv2.imwrite(os.path.join(output_folder, "object.png"), image)



    cameras = FoVPerspectiveCameras(device=device)

    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
    raster_settings = RasterizationSettings(
        image_size=(height_back, width_back),
        blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma, 
        faces_per_pixel=100, 
        perspective_correct=False,
    )

    # Create a silhouette mesh renderer by composing a rasterizer and a shader. 
    silhouette_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=SoftSilhouetteShader(blend_params=blend_params)
    )

# silhouette = silhouette_renderer(meshes_world=shapenet_model_mesh, R=R.to(device), T=T.to(device))
# silhouette = silhouette.cpu().numpy()
# silhouette_save = silhouette[0][..., 3]
# silhouette[silhouette > 0] = 1
# silhouette_path = 'silhouette.png'
# cv2.imwrite(silhouette_path, silhouette_save * 255)


    # We will save images periodically and compose them into a GIF.
    filename_output = os.path.join(output_folder, "am3s_optimization_cam.gif")
    writer = imageio.get_writer(filename_output, mode='I', duration=0.3)

    # Initialize a model using the renderer, mesh and reference image
    model = Model(meshes=shapenet_model_mesh, renderer=silhouette_renderer, image_ref=back_seg_ref, R=R, T=T).to(device)

    # Create an optimizer. Here we are using Adam and we pass in the parameters of the model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    _, image_init = model()
    image_init_save = image_init.detach().squeeze().cpu().numpy()[..., 3]
    image_init_save[image_init_save > 0] = 1
    init_path = os.path.join(output_folder, 'init.png')
    cv2.imwrite(init_path, image_init_save * 255)


    loop = tqdm(range(100))
    for i in loop:
        optimizer.zero_grad()
        loss, _ = model()
        loss.backward()
        optimizer.step()
        
        loop.set_description('Optimizing (loss %.4f)' % loss.data)
        
        # if loss.item() < 200:
        #     break
        
        # Save outputs to create a GIF. 
        if i % 10 == 0:
            R = model.R
            T = model.T

        # image = phong_renderer(meshes_world=model.meshes.clone(), R=R, T=T)
            # image = image[0, ..., :3].detach().squeeze().cpu().numpy()
            # image = img_as_ubyte(image)

            # **************** render with texture *****************
            cameras_visual = FoVPerspectiveCameras(R=R, T=T, device=device)
            raster_settings_visual = RasterizationSettings(image_size=(height_back, width_back), cull_backfaces=True)
            lights = PointLights(
                device=device, 
                location=[[0.0, 5.0, -10.0]], 
                diffuse_color=((0, 0, 0),),
                specular_color=((0, 0, 0),),
            )
            images_by_model_ids = shapenet_dataset.render(
                model_ids=[
                    model_id,
                ],
                device=device,
                cameras=cameras_visual,
                raster_settings=raster_settings_visual,
                lights=lights,
            )

            image = images_by_model_ids[0, ..., :3].detach().squeeze().cpu().numpy() * 255
            mask = images_by_model_ids[0, ..., 3].detach().squeeze().cpu().numpy()
            mask[mask > 0] = 1
            mask[mask == 0] = 0
            mask = mask[..., np.newaxis]
            combine = back_im * (1-mask) + image * mask
            combine = combine / 255
            writer.append_data(img_as_ubyte(combine))

            # image = images_by_model_ids[0, ..., :3].detach().squeeze().cpu().numpy()
            # writer.append_data(img_as_ubyte(image))
    writer.close()

    cameras_visual = FoVPerspectiveCameras(R=R, T=T, device=device)
    raster_settings_visual = RasterizationSettings(image_size=(height_back, width_back), cull_backfaces=True)
    lights = PointLights(
        device=device, 
        location=[[0.0, 5.0, -10.0]], 
        diffuse_color=((0, 0, 0),),
        specular_color=((0, 0, 0),),
    )
    images_by_model_ids = shapenet_dataset.render(
        model_ids=[
            model_id,
        ],
        device=device,
        cameras=cameras_visual,
        raster_settings=raster_settings_visual,
        lights=lights,
    )
    image = images_by_model_ids[0, ..., :3].detach().squeeze().cpu().numpy() * 255
    mask = images_by_model_ids[0, ..., 3].detach().squeeze().cpu().numpy()
    mask[mask > 0] = 1
    mask[mask == 0] = 0
    mask = mask[..., np.newaxis]
    combine = back_im * (1-mask) + image * mask
    cv2.imwrite(os.path.join(output_folder, "final.png"), cv2.cvtColor(combine, cv2.COLOR_RGB2BGR))
