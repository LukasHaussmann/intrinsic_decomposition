import argparse
import math
import os
import numpy as np
import torch
import torch.nn.functional as F

from decompose import (
    compute_sh_basis_weighted,
    predict_images,
    random_sh_lighting_2,
    alternating_least_squares,
    gradient_descent_optimizer,
    compute_albedo_scale,
    plot_results,
)


def make_checker_albedo(H, W, num_squares=8):
    """Multi-color checkerboard texture. Returns (H, W, 3) in [0, 1]."""
    palette = torch.tensor([
        [0.9, 0.2, 0.2],
        [0.2, 0.9, 0.2],
        [0.2, 0.2, 0.9],
        [0.9, 0.9, 0.2],
        [0.9, 0.2, 0.9],
        [0.2, 0.9, 0.9],
        [0.85, 0.85, 0.85],
        [0.15, 0.15, 0.15],
    ])
    sq_h = H // num_squares
    sq_w = W // num_squares
    albedo = torch.zeros(H, W, 3)
    for i in range(num_squares):
        for j in range(num_squares):
            albedo[i*sq_h:(i+1)*sq_h, j*sq_w:(j+1)*sq_w] = palette[(i * 3 + j * 5) % len(palette)]
    return albedo


def fibonacci_sphere_directions(n: int) -> list[list[float]]:
    """n evenly-distributed directions on the unit sphere via Fibonacci lattice."""
    golden = (1 + 5 ** 0.5) / 2
    dirs = []
    for i in range(n):
        phi = math.acos(1 - 2 * (i + 0.5) / n)
        theta = 2 * math.pi * i / golden
        dirs.append([math.sin(phi) * math.cos(theta),
                     math.sin(phi) * math.sin(theta),
                     math.cos(phi)])
    return dirs


def _build_sphere(device: str, specular: bool = False):
    """Build the sphere mesh and extract normals, mask, and albedo_gt via pytorch3d.
    Returns: sphere_mesh, cameras, raster_settings, black_bg, material,
             normals_flat (P,3), mask (H,W), albedo_gt (P,3)
    """
    from pytorch3d.utils import ico_sphere
    from pytorch3d.renderer import (
        FoVPerspectiveCameras, DirectionalLights, Materials,
        RasterizationSettings, MeshRenderer, MeshRasterizer,
        SoftPhongShader, TexturesUV, BlendParams, look_at_view_transform,
    )
    from pytorch3d.ops import interpolate_face_attributes

    # mesh with checker UV texture
    sphere_mesh = ico_sphere(level=3).to(device)
    verts = sphere_mesh.verts_packed()
    faces = sphere_mesh.faces_packed()
    u = (torch.atan2(verts[:, 2], verts[:, 0]) + math.pi) / (2 * math.pi)
    v = (torch.asin(verts[:, 1].clamp(-1, 1)) + math.pi / 2) / math.pi
    verts_uvs = torch.stack([u, v], dim=1)
    tex_map = make_checker_albedo(256, 256).to(device)
    sphere_mesh.textures = TexturesUV(
        maps=tex_map[None], faces_uvs=faces[None], verts_uvs=verts_uvs[None]
    )

    # camera + shared render settings
    R, T = look_at_view_transform(dist=2.7, elev=10, azim=0)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    raster_settings = RasterizationSettings(image_size=512, blur_radius=0.0, faces_per_pixel=1)
    black_bg = BlendParams(background_color=(0.0, 0.0, 0.0))
    if specular:
        material = Materials(
            device=device,
            ambient_color=((0.0, 0.0, 0.0),),
            diffuse_color=((0.7, 0.7, 0.7),),
            specular_color=((0.3, 0.3, 0.3),),
            shininess=64.0,
        )
    else:
        material = Materials(
            device=device,
            ambient_color=((0.0, 0.0, 0.0),),
            diffuse_color=((1.0, 1.0, 1.0),),
            specular_color=((0.0, 0.0, 0.0),),
            shininess=0.0,
        )

    # normals + mask: single rasterization pass
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    fragments = rasterizer(sphere_mesh)
    vertex_normals = F.normalize(verts, dim=1)
    pixel_normals = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, vertex_normals[faces]
    )[0, ..., 0, :]                                      # (H, W, 3)
    pixel_normals = F.normalize(pixel_normals, dim=-1)
    mask = (fragments.pix_to_face[0, ..., 0] >= 0)      # (H, W)
    normals_flat = pixel_normals[mask]                   # (P, 3)

    # albedo gt: ambient-only render
    albedo_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(
            device=device, cameras=cameras, blend_params=black_bg,
            lights=DirectionalLights(device=device, direction=[[0, 1.0, 0.0]],
                                     ambient_color=[[1.0, 1.0, 1.0]], diffuse_color=[[0.0, 0.0, 0.0]]),
            materials=Materials(device=device, ambient_color=((1.0, 1.0, 1.0),),
                                diffuse_color=((0.0, 0.0, 0.0),), specular_color=((0.0, 0.0, 0.0),),
                                shininess=0.0),
        ),
    )
    albedo_gt = albedo_renderer(sphere_mesh)[0, ..., :3].clamp(0, 1)[mask]  # (P, 3)

    return sphere_mesh, cameras, raster_settings, black_bg, material, normals_flat, mask, albedo_gt


def _build_cube(device: str, specular: bool = False):
    """Build a unit cube mesh with per-face flat normals and 6 solid-color faces.
    Returns the same tuple as _build_sphere.
    """
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (
        FoVPerspectiveCameras, DirectionalLights, Materials,
        RasterizationSettings, MeshRenderer, MeshRasterizer,
        SoftPhongShader, TexturesVertex, BlendParams, look_at_view_transform,
    )
    from pytorch3d.ops import interpolate_face_attributes

    # 24 unique vertices (4 per face) — CCW winding gives outward normals (verified via cross product)
    face_quads = [
        [(-0.5,-0.5, 0.5),(  0.5,-0.5, 0.5),(  0.5, 0.5, 0.5),(-0.5, 0.5, 0.5)],  # +Z
        [( 0.5,-0.5,-0.5),(-0.5,-0.5,-0.5),(-0.5, 0.5,-0.5),( 0.5, 0.5,-0.5)],   # -Z
        [(-0.5, 0.5, 0.5),(  0.5, 0.5, 0.5),(  0.5, 0.5,-0.5),(-0.5, 0.5,-0.5)],  # +Y
        [(-0.5,-0.5,-0.5),(  0.5,-0.5,-0.5),(  0.5,-0.5, 0.5),(-0.5,-0.5, 0.5)],  # -Y
        [( 0.5,-0.5, 0.5),(  0.5,-0.5,-0.5),(  0.5, 0.5,-0.5),( 0.5, 0.5, 0.5)],  # +X
        [(-0.5,-0.5,-0.5),(-0.5,-0.5, 0.5),(-0.5, 0.5, 0.5),(-0.5, 0.5,-0.5)],   # -X
    ]
    face_normals_list = [(0,0,1),(0,0,-1),(0,1,0),(0,-1,0),(1,0,0),(-1,0,0)]
    palette = [
        [0.9,0.2,0.2],[0.2,0.9,0.2],[0.2,0.2,0.9],
        [0.9,0.9,0.2],[0.9,0.2,0.9],[0.2,0.9,0.9],
    ]

    all_verts, all_colors, all_faces, tri_normals = [], [], [], []
    for i, quad in enumerate(face_quads):
        base = len(all_verts)
        all_verts.extend(quad)
        all_colors.extend([palette[i]] * 4)
        all_faces.extend([[base, base+1, base+2], [base, base+2, base+3]])
        tri_normals.extend([face_normals_list[i]] * 2)

    verts = torch.tensor(all_verts, dtype=torch.float32, device=device)
    faces = torch.tensor(all_faces, dtype=torch.int64, device=device)
    vert_colors = torch.tensor(all_colors, dtype=torch.float32, device=device)
    fn = torch.tensor(tri_normals, dtype=torch.float32, device=device)  # (12, 3)

    cube_mesh = Meshes(
        verts=[verts], faces=[faces],
        textures=TexturesVertex(verts_features=vert_colors.unsqueeze(0)),
    )

    R, T = look_at_view_transform(dist=2.7, elev=30, azim=45)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    raster_settings = RasterizationSettings(image_size=512, blur_radius=0.0, faces_per_pixel=1)
    black_bg = BlendParams(background_color=(0.0, 0.0, 0.0))
    if specular:
        material = Materials(device=device, ambient_color=((0.,0.,0.),),
                             diffuse_color=((0.7,0.7,0.7),), specular_color=((0.3,0.3,0.3),), shininess=64.)
    else:
        material = Materials(device=device, ambient_color=((0.,0.,0.),),
                             diffuse_color=((1.,1.,1.),), specular_color=((0.,0.,0.),), shininess=0.)

    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    fragments = rasterizer(cube_mesh)
    fn_per_vert = fn.unsqueeze(1).expand(-1, 3, -1).contiguous()         # (12, 3, 3)
    pixel_normals = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, fn_per_vert
    )[0, ..., 0, :]                                                       # (H, W, 3)
    pixel_normals = F.normalize(pixel_normals, dim=-1)
    mask = (fragments.pix_to_face[0, ..., 0] >= 0)                       # (H, W)
    normals_flat = pixel_normals[mask]                                    # (P, 3)

    albedo_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(
            device=device, cameras=cameras, blend_params=black_bg,
            lights=DirectionalLights(device=device, direction=[[0,1.,0.]],
                                     ambient_color=[[1.,1.,1.]], diffuse_color=[[0.,0.,0.]]),
            materials=Materials(device=device, ambient_color=((1.,1.,1.),),
                                diffuse_color=((0.,0.,0.),), specular_color=((0.,0.,0.),), shininess=0.),
        ),
    )
    albedo_gt = albedo_renderer(cube_mesh)[0, ..., :3].clamp(0, 1)[mask]  # (P, 3)

    return cube_mesh, cameras, raster_settings, black_bg, material, normals_flat, mask, albedo_gt


def _build_mesh(device: str, specular: bool = False, shape: str = "sphere"):
    if shape == "cube":
        return _build_cube(device, specular=specular)
    return _build_sphere(device, specular=specular)


def _random_point_lights(num_lights: int, radius: float = 3.0, seed: int = None, device: str = "cpu", specular: bool = False):
    """Sample num_lights point lights with random positions and colors.
    Returns list of PointLights, one per light source.
    """
    from pytorch3d.renderer import PointLights
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)
    dirs = F.normalize(torch.randn(num_lights, 3, generator=rng), dim=1)
    # flip lights that face away from the camera (camera is at roughly (0, 0.47, 2.66))
    cam_dir = F.normalize(torch.tensor([0.0, 0.47, 2.66]), dim=0)
    dirs[dirs @ cam_dir < 0] *= -1
    positions = dirs * radius
    colors = torch.rand(num_lights, 3, generator=rng).clamp(0.3, 1.0)
    spec = [[1.0, 1.0, 1.0]] if specular else [[0, 0, 0]]
    return [
        PointLights(
            device=device,
            location=[positions[i].tolist()],
            ambient_color=[[0, 0, 0]],
            diffuse_color=[colors[i].tolist()],
            specular_color=spec,
        )
        for i in range(num_lights)
    ]


def get_data_pytorch3d(num_images: int, device: str, lights_per_image: int = 4, seed: int = 0, specular: bool = False, shape: str = "sphere"):
    """Render sphere/cube under num_images complex lighting conditions using pytorch3d.
    Each image uses lights_per_image random colored point lights, producing high-frequency
    shading that exceeds what order-2 SH can represent.
    With specular=True the material has a specular component, further violating the Lambertian assumption.
    Returns: images_flat (K,P,3), normals_flat (P,3), mask (H,W), albedo_gt (P,3)
    """
    from pytorch3d.renderer import PointLights, MeshRenderer, MeshRasterizer, SoftPhongShader

    mesh, cameras, raster_settings, black_bg, material, normals_flat, mask, albedo_gt = _build_mesh(device, specular=specular, shape=shape)

    dummy_lights = PointLights(device=device, location=[[0, 0, -3]],
                               ambient_color=[[0, 0, 0]], diffuse_color=[[1, 1, 1]], specular_color=[[0, 0, 0]])
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=dummy_lights,
                               materials=material, blend_params=black_bg),
    )

    imgs_flat = []
    for k in range(num_images):
        # each image gets independent random light positions and colors
        point_lights = _random_point_lights(lights_per_image, device=device, seed=seed + k, specular=specular)
        img = sum(renderer(mesh, lights=pl)[0, ..., :3] for pl in point_lights)
        imgs_flat.append(img.clamp(0, 1)[mask])
    images_flat = torch.stack(imgs_flat, dim=0)          # (K, P, 3)

    return images_flat, normals_flat, mask, albedo_gt


def get_data_sh_synth(num_images: int, device: str, shape: str = "sphere"):
    """Render sphere/cube under random SH lighting using the SH forward model.
    Geometry (normals/mask/albedo_gt) comes from a single pytorch3d pass;
    images are generated analytically from SH coefficients.
    Returns: images_flat (K,P,3), normals_flat (P,3), mask (H,W), albedo_gt (P,3)
    """
    _, _, _, _, _, normals_flat, mask, albedo_gt = _build_mesh(device, shape=shape)

    y_tilde = compute_sh_basis_weighted(normals_flat)
    L = random_sh_lighting_2(num_images).to(device)
    with torch.no_grad():
        images_flat = predict_images(albedo_gt, L, y_tilde).clamp(0, 1)

    return images_flat, normals_flat, mask, albedo_gt


def get_data_from_files(data_dir: str, device: str):
    """Load renders + normals + albedo from a synthetic_pytorch3d_dataset sample dir.
    Mask is derived from the normal map: background pixels are pure black (0,0,0).
    Returns: images_flat (K,P,3), normals_flat (P,3), mask (H,W), albedo_gt (P,3)
    """
    import glob
    from PIL import Image

    def load_png(path):
        return torch.from_numpy(
            np.array(Image.open(path)).astype(np.float32) / 255.0
        ).to(device)

    # normals: encoded as (n+1)/2 in [0,1], background is pure black
    normal_img = load_png(os.path.join(data_dir, "normal.png"))   # (H,W,4)
    normal_rgb = normal_img[..., :3]
    mask = normal_rgb.sum(-1) > 0                                   # (H,W)
    normals_flat = normal_rgb[mask] * 2.0 - 1.0                    # decode to [-1,1]
    normals_flat = F.normalize(normals_flat, dim=-1)               # (P,3)

    # albedo ground truth
    albedo_img = load_png(os.path.join(data_dir, "albedo.png"))    # (H,W,4)
    albedo_gt = albedo_img[..., :3][mask]                           # (P,3)

    # RGB renders sorted by filename
    rgb_paths = sorted(glob.glob(os.path.join(data_dir, "rgb_*.png")))
    imgs_flat = []
    for path in rgb_paths:
        img = load_png(path)[..., :3]                              # (H,W,3)
        imgs_flat.append(img[mask])
    images_flat = torch.stack(imgs_flat, dim=0)                    # (K,P,3)

    return images_flat, normals_flat, mask, albedo_gt


def main():
    parser = argparse.ArgumentParser(description="Intrinsic decomposition")
    parser.add_argument("--mode", choices=["pytorch3d", "sh_synth", "from_files"], default="sh_synth")
    parser.add_argument("--method", choices=["als", "gradient"], default="gradient")
    parser.add_argument("--num_images", type=int, default=8)
    parser.add_argument("--data_dir", default="data/00000", help="only used with --mode from_files")
    parser.add_argument("--specular", action="store_true", help="use specular material (pytorch3d mode only)")
    parser.add_argument("--shape", choices=["sphere", "cube"], default="sphere", help="geometry for pytorch3d/sh_synth modes")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- data ---
    if args.mode == "pytorch3d":
        images_flat, normals_flat, mask, albedo_gt = get_data_pytorch3d(args.num_images, device, specular=args.specular, shape=args.shape)
    elif args.mode == "sh_synth":
        images_flat, normals_flat, mask, albedo_gt = get_data_sh_synth(args.num_images, device, shape=args.shape)
    else:
        images_flat, normals_flat, mask, albedo_gt = get_data_from_files(args.data_dir, device)

    # --- decompose ---
    if args.method == "als":
        rho, L = alternating_least_squares(normals_flat, images_flat)
    else:
        rho, L = gradient_descent_optimizer(normals_flat, images_flat, mask, num_steps=2000)

    # --- evaluate ---
    y_tilde   = compute_sh_basis_weighted(normals_flat)
    predicted = predict_images(rho, L, y_tilde)
    loss      = ((predicted - images_flat) ** 2).mean()
    print(f"Final loss: {loss:.6f}")

    alpha = None
    if albedo_gt is not None:
        alpha = compute_albedo_scale(rho, albedo_gt)
        mae   = ((rho * alpha - albedo_gt)).abs().mean()
        print(f"Albedo MAE (rescaled): {mae:.6f}")
        print(f"Per-channel scale factors α: {alpha.tolist()}")

    plot_results(
        rho_gt=albedo_gt, rho_unaligned=rho, alpha=alpha,
        images_flat=images_flat, predicted=predicted,
        y_tilde=y_tilde, L_pred=L,
        mask=mask, num_show=args.num_images,
        path="results.png",
    )


if __name__ == "__main__":
    main()
