import glob
import os
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image


"""
    lambertian + SH intrinsic decomposition
    p : pixel, A(p) : albedo of pixel p, S(p) : shading of pixel p
    I(p) = A(p) * S(p)
    S(p) : integral over light angles omega that are above the upper hemisphere w.r.t the normal n of p
    L := SH lighting coefficients
    
    Rendering equation:
    I(p) = A(p) * 
"""
# compute weighted SH basis functions
def compute_sh_basis_weighted(normals: torch.Tensor) -> torch.Tensor:
    x = normals[..., 0]
    y = normals[..., 1]
    z = normals[..., 2]

    pi = math.pi
    c0 = 0.5 * math.sqrt(pi)              # band 0
    c1 = (pi / 3.0) * math.sqrt(3.0 / pi) # band 1
    c2 = (pi / 8.0) * math.sqrt(15.0 / pi) # band 2, m = ±2 off-diag and m = ±1
    c3 = (pi / 16.0) * math.sqrt(5.0 / pi) # band 2, m = 0
    c4 = (pi / 16.0) * math.sqrt(15.0 / pi) # band 2, m = +2
    # torch.stack along a new last dim gives shape (..., 9)
    y_tilde = torch.stack([
        c0 * torch.ones_like(x),  # i=0:  1
        c1 * y,                   # i=1:  y
        c1 * z,                   # i=2:  z
        c1 * x,                   # i=3:  x
        c2 * x * y,               # i=4:  xy
        c2 * y * z,               # i=5:  yz
        c3 * (3.0 * z * z - 1.0), # i=6:  3z² - 1
        c2 * x * z,               # i=7:  xz
        c4 * (x * x - y * y),     # i=8:  x² - y²
    ], dim=-1)

    return y_tilde

# init parameters: sh coefficients + albedo map
def init_parameters(num_images: int, num_pixels: int, device='cuda'):
    # Lighting: (K, 3 channels, 9 SH coefficients)
    # Initialize to a mild ambient: DC term ≈ 1, rest 0
    L = torch.zeros(num_images, 3, 9, device=device)
    L[:, :, 0] = 1.0
    L = torch.nn.Parameter(L)

    # Albedo: (P, 3 channels)
    rho = torch.full((num_pixels, 3), 0.5, device=device)
    rho = torch.nn.Parameter(rho)

    return L, rho

def predict_images(rho: torch.Tensor, L: torch.Tensor, y_tilde: torch.Tensor) -> torch.Tensor:
    """
    Args:
        rho:     (P, 3)       per-pixel albedo
        L:       (K, 3, 9)    per-image, per-channel SH coeffs
        y_tilde: (P, 9)       cosine-weighted basis at each pixel

    Returns:
        (K, P, 3) predicted intensities
    """
    # Irradiance per (image, pixel, channel):
    #   E[k, p, c] = sum_i  y_tilde[p, i] * L[k, c, i]
    # einsum is the cleanest way to express this.
    irradiance = torch.einsum('pi,kci->kpc', y_tilde, L)

    # Multiply by albedo / pi
    # rho is (P, 3); broadcast across K dim
    predicted = (rho.unsqueeze(0) / math.pi) * irradiance

    return predicted

def solve_lighting(rho: torch.Tensor,
                   y_tilde: torch.Tensor,
                   images_flat: torch.Tensor) -> torch.Tensor:
    """
    Args:
        rho:         (P, 3)
        y_tilde:     (P, 9)
        images_flat: (K, P, 3)

    Returns:
        L: (K, 3, 9)
    """
    P = y_tilde.shape[0]
    K = images_flat.shape[0]
    pi = math.pi

    # Build per-channel design matrices: A_c[p, i] = (rho[p, c] / pi) * y_tilde[p, i]
    # Shape: (3, P, 9) — one design matrix per channel.
    A = (rho.t().unsqueeze(-1) / pi) * y_tilde.unsqueeze(0)

    # Reshape observations to (3, K, P) — group by channel for batching with A.
    # images_flat is (K, P, 3); permute to (3, K, P).
    b = images_flat.permute(2, 0, 1)  # (3, K, P)
    # We want to solve: for each channel c, A[c] @ L[c, k] = b[c, k]
    # That's 3 channels × K images = 3K systems, all sharing A[c] within a channel.

    # torch.linalg.lstsq broadcasts. We need:
    #   A: (3, P, 9)         — broadcasts to (3, 1, P, 9)
    #   b: (3, K, P, 1)      — last dim is the RHS column
    # Output: (3, K, 9, 1)
    A_b = A.unsqueeze(1).expand(3, K, P, 9)        # (3, K, P, 9)
    b_b = b.unsqueeze(-1)                           # (3, K, P, 1)

    solution = torch.linalg.lstsq(A_b, b_b).solution  # (3, K, 9, 1)

    # Squeeze the trailing 1 and permute to (K, 3, 9)
    L = solution.squeeze(-1).permute(1, 0, 2)

    return L

def solve_albedo(L: torch.Tensor,
                 y_tilde: torch.Tensor,
                 images_flat: torch.Tensor) -> torch.Tensor:
    """
    Args:
        L:           (K, 3, 9)
        y_tilde:     (P, 9)
        images_flat: (K, P, 3)

    Returns:
        rho: (P, 3)
    """
    pi = math.pi
    # Per-image, per-pixel, per-channel scale: s[k, p, c] = (1/pi) * y_tilde[p, :] . L[k, c, :]
    s = torch.einsum('pi,kci->kpc', y_tilde, L) / pi  # (K, P, 3)

    # 1D least squares: rho[p, c] = sum_k s[k, p, c] * I[k, p, c] / sum_k s[k, p, c]^2
    numerator   = (s * images_flat).sum(dim=0)        # (P, 3)
    denominator = (s * s).sum(dim=0)                  # (P, 3)

    # Guard against divide-by-zero where lighting cancels out at a pixel
    eps = 1e-8
    rho = numerator / (denominator + eps)

    return rho

def load_render_data(
    image_paths,
    normal_map_path="normal_map.png",
    mask_path="mask.png",
    device="cuda",
):
    """
    Args:
        image_paths: str or list of str — one or more rendered image paths
    Returns:
        images_flat: (K, P, 3) float32 in [0, 1]
        normals_flat: (P, 3) float32 in [-1, 1]
        mask:         (H, W) bool
    """
    if isinstance(image_paths, str) and os.path.isdir(image_paths):
        image_paths = sorted(glob.glob(os.path.join(image_paths, "*.png")))
    elif isinstance(image_paths, str):
        image_paths = [image_paths]

    normals = np.array(Image.open(normal_map_path)).astype(np.float32) / 255.0 * 2.0 - 1.0
    mask = np.array(Image.open(mask_path)).astype(bool)

    normals_t = torch.from_numpy(normals).to(device)
    mask_t = torch.from_numpy(mask).to(device)
    normals_flat = normals_t[mask_t]  # (P, 3)

    imgs_flat = []
    for path in image_paths:
        img = np.array(Image.open(path)).astype(np.float32) / 255.0
        img_t = torch.from_numpy(img).to(device)
        imgs_flat.append(img_t[mask_t])  # (P, 3)

    images_flat = torch.stack(imgs_flat, dim=0)  # (K, P, 3)

    return images_flat, normals_flat, mask_t

def save_predicted_images(predicted: torch.Tensor, mask: torch.Tensor, output_dir: str = "."):
    """
    Args:
        predicted:  (K, P, 3) predicted intensities in [0, 1]
        mask:       (H, W) bool
        output_dir: directory to save images to
    """
    H, W = mask.shape
    K = predicted.shape[0]
    pred_np = predicted.detach().cpu().numpy()
    mask_np = mask.cpu().numpy()

    for k in range(K):
        canvas = np.zeros((H, W, 3), dtype=np.float32)
        canvas[mask_np] = pred_np[k]
        img = (np.clip(canvas, 0, 1) * 255).astype(np.uint8)
        os.makedirs(output_dir, exist_ok=True)
        Image.fromarray(img).save(f"{output_dir}/predicted_{k:04d}.png")

def save_shading_maps(L: torch.Tensor, y_tilde: torch.Tensor, mask: torch.Tensor, output_dir: str = "."):
    """
    Args:
        L:       (K, 3, 9) SH lighting coefficients
        y_tilde: (P, 9) cosine-weighted SH basis
        mask:    (H, W) bool
    """
    shading = torch.einsum('pi,kci->kpc', y_tilde, L) / math.pi  # (K, P, 3)
    H, W = mask.shape
    K = shading.shape[0]
    shading_np = shading.detach().cpu().numpy()
    mask_np = mask.cpu().numpy()

    os.makedirs(output_dir, exist_ok=True)
    for k in range(K):
        canvas = np.zeros((H, W, 3), dtype=np.float32)
        canvas[mask_np] = shading_np[k]
        img = (np.clip(canvas, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img).save(f"{output_dir}/shading_{k:04d}.png")


def save_albedo(rho: torch.Tensor, mask: torch.Tensor, path: str = "albedo.png"):
    """
    Args:
        rho:  (P, 3) albedo in [0, 1]
        mask: (H, W) bool
    """
    H, W = mask.shape
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    canvas[mask.cpu().numpy()] = rho.detach().cpu().numpy()
    img = (np.clip(canvas, 0, 1) * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Image.fromarray(img).save(path)

def random_sh_lighting(num_images=4, seed=None):
    """
    Generate random SH coefficients with realistic frequency falloff.
    Returns (K, 3, 9).
    """
    if seed is not None:
        torch.manual_seed(seed)

    K = num_images
    L = torch.zeros(K, 3, 9)

    # Band-wise std: lower bands are dominant
    band_scale = torch.tensor([
        1.0,                      # l=0 (DC)
        0.5, 0.5, 0.5,            # l=1
        0.25, 0.25, 0.25, 0.25, 0.25  # l=2
    ])

    for k in range(K):
        # Sample a base "color temperature" for this image
        # so different images have visibly different lighting
        color_bias = 0.7 + 0.6 * torch.rand(3)   # per-channel scale, [0.7, 1.3]

        for c in range(3):
            # Gaussian samples scaled by band falloff
            coeffs = torch.randn(9) * band_scale

            # Force the DC term to be positive and dominant
            # so the lighting is mostly positive everywhere
            coeffs[0] = (1.5 + 0.5 * torch.rand(1).item()) * color_bias[c]

            L[k, c] = coeffs

    return L

def random_sh_lighting_2(num_images=4, seed=None):
    """
    Generate diverse SH coefficients across images.
    Returns (K, 3, 9).
    """
    if seed is not None:
        torch.manual_seed(seed)

    K = num_images
    L = torch.zeros(K, 3, 9)

    # Per-band magnitudes — keep band 1 large enough to dominate the
    # variation across images.
    band_std = torch.tensor([
        0.0,                       # DC handled separately
        0.8, 0.8, 0.8,             # l=1: strong directional component
        0.3, 0.3, 0.3, 0.3, 0.3,   # l=2: smaller but nontrivial
    ])

    for k in range(K):
        color_bias = 0.7 + 0.6 * torch.rand(3)  # per-channel chromatic bias

        # Pick a dominant lighting direction for this image — uniformly on the sphere.
        # This ensures different images have lighting from genuinely different directions.
        direction = torch.randn(3)
        direction = direction / direction.norm()

        for c in range(3):
            coeffs = torch.randn(9) * band_std

            # DC: positive, but only modestly larger than band-1 magnitudes
            coeffs[0] = (1.0 + 0.3 * torch.rand(1).item()) * color_bias[c]

            # Inject the chosen direction into band 1.
            # Band-1 SH order is (y, z, x), so map direction (x, y, z) accordingly.
            band1_strength = 0.7 + 0.4 * torch.rand(1).item()
            coeffs[1] = band1_strength * direction[1] * color_bias[c]  # y
            coeffs[2] = band1_strength * direction[2] * color_bias[c]  # z
            coeffs[3] = band1_strength * direction[0] * color_bias[c]  # x

            L[k, c] = coeffs

    return L

def alternating_least_squares(normals_flat, images_flat,
                              num_iters=30, verbose=True):
    P = normals_flat.shape[0]
    K = images_flat.shape[0]
    device = normals_flat.device

    y_tilde = compute_sh_basis_weighted(normals_flat)  # (P, 9)

    #init_color = torch.tensor([0.8, 0.5, 0.2], device=device)  # (3,)
    #rho = init_color.expand(P, 3).clone()                       # (P, 3)

    # Initialize albedo to mid-gray
    rho = torch.full((P, 3), 0.7, device=device)
    prev_loss = float('inf')
    for it in range(num_iters):
        L = solve_lighting(rho, y_tilde, images_flat)
        rho = solve_albedo(L, y_tilde, images_flat)

        # Optional: clip albedo to [0, 1] to handle scale ambiguity
        #rho = rho.clamp(1e-6, 1.0)

        # Check convergence
        with torch.no_grad():
            predicted = predict_images(rho, L, y_tilde)
            loss = ((predicted - images_flat) ** 2).mean().item()
            if verbose:
                print(f"Iter {it}: loss = {loss:.6f}")
            #if abs(prev_loss - loss) < 1e-7:
            #    break
            prev_loss = loss

    return rho, L

def total_variation_loss(rho, H, W, mask):
    """L1 total variation of albedo. rho is (P, 3), reshape to (H, W, 3)."""
    rho_image = torch.zeros(H, W, 3, device=rho.device)
    rho_image[mask] = rho

    # Horizontal and vertical differences
    dh = (rho_image[:, 1:] - rho_image[:, :-1]).abs()
    dv = (rho_image[1:, :] - rho_image[:-1, :]).abs()

    return dh.sum() + dv.sum()

def init_lighting_ambient(K, device='cuda'):
    """Initialize K lights as uniform white ambient."""
    L = torch.zeros(K, 3, 9, device=device)
    L[:, :, 0] = 1.0  # band-0 (DC) per channel, per image
    return L

def gradient_descent_optimizer(normals_flat, images_flat, mask,
                               num_steps=30, verbose=True):
    P = normals_flat.shape[0]
    K = images_flat.shape[0]
    device = normals_flat.device

    H = mask.shape[0]
    W = mask.shape[1]
    print(H, W)

    # Initialize albedo to mid-gray
    rho_param = torch.nn.Parameter(torch.full((P, 3), 0.7, device=device))
    L_param = torch.nn.Parameter(init_lighting_ambient(K, device=device))

    y_tilde = compute_sh_basis_weighted(normals_flat)  # (P, 9)

    optimizer = torch.optim.Adam([rho_param, L_param], lr=0.01)

    for step in range(num_steps):
        optimizer.zero_grad()
        predicted = predict_images(rho_param, L_param, y_tilde)
        recon_loss = ((predicted - images_flat) ** 2).mean()
        tv_loss = total_variation_loss(rho_param, H, W, mask)
        loss = recon_loss #+ 1e-9 * tv_loss
        loss.backward()
        optimizer.step()
        rho_param.data.clamp_(0, 1)
        if verbose:
            print(f"Iter {step}: loss = {loss:.6f}")

    return rho_param, L_param

def generate_images(
    albedo_gt_path: str = "albedo_gt.png",
    normal_map_path: str = "normal_map.png",
    mask_path: str = "mask.png",
    num_variations: int = 4,
    output_dir: str = "renders_rand",
    shading_dir: str = "shadings_rand",
    device: str = "cuda",
):
    """Render num_variations images under random SH lighting from ground-truth albedo."""
    normals = np.array(Image.open(normal_map_path)).astype(np.float32) / 255.0 * 2.0 - 1.0
    mask    = np.array(Image.open(mask_path)).astype(bool)
    albedo  = np.array(Image.open(albedo_gt_path)).astype(np.float32) / 255.0

    mask_t    = torch.from_numpy(mask).to(device)
    normals_t = torch.from_numpy(normals).to(device)
    albedo_t  = torch.from_numpy(albedo).to(device)

    normals_flat = normals_t[mask_t]
    rho_flat     = albedo_t[mask_t]

    y_tilde = compute_sh_basis_weighted(normals_flat)
    L = random_sh_lighting_2(num_images=num_variations, seed=1).to(device)
    with torch.no_grad():
        images_flat = predict_images(rho_flat, L, y_tilde).clamp(0.0, 1.0)

    H, W = mask_t.shape
    mask_np   = mask_t.cpu().numpy()
    images_np = images_flat.cpu().numpy()
    os.makedirs(output_dir, exist_ok=True)
    for k in range(num_variations):
        canvas = np.zeros((H, W, 3), dtype=np.float32)
        canvas[mask_np] = images_np[k]
        Image.fromarray((canvas.clip(0, 1) * 255).astype(np.uint8)).save(
            f"{output_dir}/render_{k:04d}.png"
        )
    save_shading_maps(L, y_tilde, mask_t, output_dir=shading_dir)

def compute_albedo_scale(rho_est, rho_gt):
    """
    rho_est, rho_gt: (P, 3)
    Returns: (3,) per-channel scale factors.
    """
    numerator   = (rho_gt * rho_est).sum(dim=0)   # (3,)
    denominator = (rho_est * rho_est).sum(dim=0)  # (3,)
    return numerator / denominator

def plot_results(
    rho_gt: torch.Tensor,
    rho_unaligned: torch.Tensor,
    images_flat: torch.Tensor,
    predicted: torch.Tensor,
    y_tilde: torch.Tensor,
    L_pred: torch.Tensor,
    mask: torch.Tensor,
    shading_gt_dir: str = None,
    num_show: int = 4,
    path: str = "results.png",
):
    """
    Layout: one left column with GT albedo (top half) and estimated albedo (bottom half),
    each ~4x larger than a render cell. Right 4 columns have num_show rows of
    GT Render | Reconstructed Render | GT Shading | Estimated Shading.
    """
    H, W = mask.shape
    mask_np = mask.cpu().numpy()
    num_show = min(num_show, images_flat.shape[0])

    pred_shading = (torch.einsum('pi,kci->kpc', y_tilde, L_pred) / math.pi).clamp(0, 1)

    def to_img(flat):
        canvas = np.zeros((H, W, 3), dtype=np.float32)
        canvas[mask_np] = flat.detach().cpu().float().numpy()
        return canvas.clip(0, 1)

    # Albedo cells are 2 rows tall and 2 cols wide → 4× the area of a render cell.
    # Total grid: num_show rows × 6 cols, width_ratios=[2,1,1,1,1] (albedo col = 2 units wide).
    cell = 3  # inches per render cell
    half = num_show // 2
    gs = gridspec.GridSpec(num_show, 5, hspace=0.05, wspace=0.08,
                           width_ratios=[2, 1, 1, 1, 1])
    fig = plt.figure(figsize=(cell * (2 + 4), cell * num_show))

    # --- albedo column (col 0): GT top half, estimated bottom half ---
    ax_gt  = fig.add_subplot(gs[:half, 0])
    ax_est = fig.add_subplot(gs[half:, 0])
    ax_gt.imshow(to_img(rho_gt));         ax_gt.axis('off')
    ax_est.imshow(to_img(rho_unaligned)); ax_est.axis('off')
    ax_gt.set_title('GT Albedo',        fontsize=11, pad=6)
    ax_est.set_title('Estimated Albedo', fontsize=11, pad=6)

    # --- per-image columns (cols 1-4) ---
    for row in range(num_show):
        axes = [fig.add_subplot(gs[row, c]) for c in range(1, 5)]
        axes[0].imshow(to_img(images_flat[row]))
        axes[1].imshow(to_img(predicted[row].detach()))
        if shading_gt_dir is not None:
            gt_shade = np.array(
                Image.open(f"{shading_gt_dir}/shading_{row:04d}.png")
            ).astype(np.float32) / 255.0
            axes[2].imshow(gt_shade.clip(0, 1))
        axes[3].imshow(to_img(pred_shading[row]))
        for ax in axes:
            ax.axis('off')
        if row == 0:
            for ax, title in zip(axes, ['GT Rendering', 'Reconstructed Rendering',
                                        'GT Shading', 'Estimated Shading']):
                ax.set_title(title, fontsize=11, pad=6)

    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved results figure to {path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    generate_images(num_variations=8,device=device)

    images_flat, normals_flat, mask = load_render_data("renders_rand", device=device)
    #rho, L = alternating_least_squares(normals_flat, images_flat, num_iters=100)
    rho, L = gradient_descent_optimizer(normals_flat, images_flat, mask, num_steps=200)

    y_tilde   = compute_sh_basis_weighted(normals_flat)
    predicted = predict_images(rho, L, y_tilde)
    loss      = ((predicted - images_flat) ** 2).mean()
    print(f"Final loss: {loss:.6f}")

    albedo_gt  = torch.from_numpy(np.array(Image.open('albedo_gt.png')).astype(np.float32) / 255.0).to(device)
    rho_gt = albedo_gt[mask]

    alpha = compute_albedo_scale(rho, rho_gt)
    print(alpha)
    rho_aligned = rho * alpha.unsqueeze(0)  # (P, 3)
    # How well does the rescaled albedo match?
    albedo_residual = (rho_aligned - rho_gt).abs()
    print(f"Albedo MAE after per-channel rescaling: {albedo_residual.mean():.6f}")
    print(f"Albedo max abs error:                   {albedo_residual.max():.6f}")
    print(f"Per-channel scale factors α: {alpha.tolist()}")

    save_predicted_images(predicted, mask, output_dir='predicted_renders')
    save_albedo(rho, mask)
    save_albedo(rho_aligned, mask, path='albedo_aligned.png')
    save_shading_maps(L, y_tilde, mask, output_dir='predicted_shadings')

    plot_results(
        rho_gt=rho_gt, rho_unaligned=rho,
        images_flat=images_flat, predicted=predicted,
        y_tilde=y_tilde, L_pred=L,
        mask=mask, shading_gt_dir='shadings_rand',
        num_show=8, path='results.png',
    )

if __name__ == "__main__":
    main()