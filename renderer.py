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


def make_checker_albedo(H, W, num_squares=8):
    """Multi-color checkerboard. Returns (H, W, 3) in [0, 1]."""
    palette = torch.tensor([
        [0.9, 0.2, 0.2],   # red
        [0.2, 0.9, 0.2],   # green
        [0.2, 0.2, 0.9],   # blue
        [0.9, 0.9, 0.2],   # yellow
        [0.9, 0.2, 0.9],   # magenta
        [0.2, 0.9, 0.9],   # cyan
        [0.85, 0.85, 0.85],# near-white
        [0.15, 0.15, 0.15],# near-black
    ])
    sq_h = H // num_squares
    sq_w = W // num_squares
    albedo = torch.zeros(H, W, 3)
    for i in range(num_squares):
        for j in range(num_squares):
            color_idx = (i * 3 + j * 5) % len(palette)
            albedo[i*sq_h:(i+1)*sq_h, j*sq_w:(j+1)*sq_w] = palette[color_idx]
    return albedo


ALBEDO_IMAGE_PATH = "sheep.jpg"

sphere_mesh = ico_sphere(level=3).to(device)
verts = sphere_mesh.verts_packed()
faces = sphere_mesh.faces_packed()

x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
u = (torch.atan2(z, x) + torch.pi) / (2 * torch.pi)
v = (torch.asin(y.clamp(-1, 1)) + torch.pi / 2) / torch.pi
verts_uvs = torch.stack([u, v], dim=1).to(device)  # (V, 2)

tex_map = make_checker_albedo(256, 256).to(device)  # (H, W, 3)
#tex_map = torch.tensor([0.7, 0.6, 0.2], device=device).expand(256, 256, 3).contiguous()  # single color
#tex_map = torch.from_numpy(
#    __import__("numpy").array(Image.open(ALBEDO_IMAGE_PATH).convert("RGB"), dtype="float32") / 255.0
#).to(device)
sphere_mesh.textures = TexturesUV(
    maps=tex_map[None],           # (1, H, W, 3)
    faces_uvs=faces[None],        # (1, F, 3)
    verts_uvs=verts_uvs[None],    # (1, V, 2)
)


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