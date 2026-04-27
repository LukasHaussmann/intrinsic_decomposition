import os
import sys
import torch
import matplotlib.pyplot as plt

# Util function for loading meshes
from PIL import Image

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,
    BlendParams,
)
from pytorch3d.ops import interpolate_face_attributes


# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


sphere_mesh = ico_sphere(level=3).to(device)
verts = sphere_mesh.verts_packed()
verts_rgb = torch.ones_like(verts)[None] * torch.tensor(
    [0, 0, 0.9], device=device
)
sphere_mesh.textures = TexturesVertex(verts_features=verts_rgb)


R, T = look_at_view_transform(dist=2.7, elev=10, azim=0)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

raster_settings = RasterizationSettings(image_size=512, blur_radius=0.0, faces_per_pixel=1)
lights = DirectionalLights(
    device=device,
    direction=[[0, 1.0, 1.0]],
    ambient_color=[[0, 0, 0]],
    diffuse_color=[[1.0, 1.0, 1.0]],
)

diffuse_material = Materials(
    device=device,
    ambient_color=((0.0, 0.0, 0.0),),
    diffuse_color=((1.0, 1.0, 1.0),),
    specular_color=((0.0, 0.0, 0.0),),
    shininess=0.0,
)
black_background = BlendParams(background_color=(0.0, 0.0, 0.0))
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    shader=SoftPhongShader(device=device, cameras=cameras, lights=lights, materials=diffuse_material, blend_params=black_background),
)


os.makedirs("renders", exist_ok=True)

light_directions = [
    [1.0,  0.0,  0.0],
    [-1.0, 0.0,  0.0],
    [0.0,  1.0,  0.0],
    [0.0, -1.0,  0.0],
    [0.0,  0.0,  1.0],
    [0.0,  0.0, -1.0],
    [1.0,  1.0,  1.0],
    [-1.0, 1.0,  1.0],
]

for k, direction in enumerate(light_directions):
    lights_k = DirectionalLights(
        device=device,
        direction=[direction],
        ambient_color=[[0, 0, 0]],
        diffuse_color=[[1.0, 1.0, 1.0]],
    )
    images = renderer(sphere_mesh, lights=lights_k)
    img = (images[0, ..., :3].cpu().numpy() * 255).astype("uint8")
    Image.fromarray(img).save(f"renders/render_{k:04d}.png")

# Get normal map
rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
fragments = rasterizer(sphere_mesh)

# For a sphere, vertex normals = normalized vertex positions
vertex_normals = torch.nn.functional.normalize(verts, dim=1)  # (V, 3)
faces = sphere_mesh.faces_packed()  # (F, 3)
face_vertex_normals = vertex_normals[faces]  # (F, 3, 3)

# Interpolate normals at each pixel
pixel_normals = interpolate_face_attributes(
    fragments.pix_to_face,
    fragments.bary_coords,
    face_vertex_normals
)  # (1, H, W, 1, 3)
pixel_normals = pixel_normals[0, ..., 0, :]  # (H, W, 3)
pixel_normals = torch.nn.functional.normalize(pixel_normals, dim=-1)

# Map [-1, 1] to [0, 1] for visualization
normal_map = (pixel_normals + 1) / 2
normal_map = (normal_map.cpu().numpy() * 255).astype("uint8")
Image.fromarray(normal_map).save("normal_map.png")

# Object mask: True where a face was rasterized
mask = (fragments.pix_to_face[0, ..., 0] >= 0)  # (H, W)
mask_img = (mask.cpu().numpy().astype("uint8") * 255)
Image.fromarray(mask_img).save("mask.png")

# Albedo ground truth: ambient-only render so output = raw texture color
albedo_lights = DirectionalLights(
    device=device,
    direction=[[0, 1.0, 0.0]],
    ambient_color=[[1.0, 1.0, 1.0]],
    diffuse_color=[[0.0, 0.0, 0.0]],
)
albedo_material = Materials(
    device=device,
    ambient_color=((1.0, 1.0, 1.0),),
    diffuse_color=((0.0, 0.0, 0.0),),
    specular_color=((0.0, 0.0, 0.0),),
    shininess=0.0,
)
albedo_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    shader=SoftPhongShader(device=device, cameras=cameras, lights=albedo_lights, materials=albedo_material, blend_params=black_background),
)
albedo_img = albedo_renderer(sphere_mesh)
albedo_img = (albedo_img[0, ..., :3].cpu().numpy() * 255).astype("uint8")
Image.fromarray(albedo_img).save("albedo_gt.png")