import glob
import os
import torch
import math
import numpy as np
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

def generate_lighting_variations(
    rho: torch.Tensor,
    normals_flat: torch.Tensor,
    num_variations: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        rho:          (P, 3) base albedo in [0, 1]
        normals_flat: (P, 3) surface normals in [-1, 1]
        num_variations: number of lighting conditions to generate

    Returns:
        images_flat: (K, P, 3) rendered images, clamped to [0, 1]
        L:           (K, 3, 9) the random SH coefficients used
    """
    device = rho.device
    K = num_variations

    y_tilde = compute_sh_basis_weighted(normals_flat)  # (P, 9)

    L = torch.randn(K, 3, 9, device=device)
    L[:, :, 0] = L[:, :, 0].abs()          # DC term must be positive (net light energy)
    L[:, :, 1:4] *= 0.6                    # band-1 directional terms smaller
    L[:, :, 4:] *= 0.3                     # band-2 terms smaller still

    with torch.no_grad():
        images_flat = predict_images(rho, L, y_tilde).clamp(0.0, 1.0)  # (K, P, 3)

    return images_flat, L


def alternating_least_squares(normals_flat, images_flat,
                              num_iters=30, verbose=True):
    P = normals_flat.shape[0]
    K = images_flat.shape[0]
    device = normals_flat.device

    y_tilde = compute_sh_basis_weighted(normals_flat)  # (P, 9)

    # Initialize albedo to mid-gray
    rho = torch.full((P, 3), 0.5, device=device)
    prev_loss = float('inf')
    for it in range(num_iters):
        L = solve_lighting(rho, y_tilde, images_flat)
        rho = solve_albedo(L, y_tilde, images_flat)

        # Optional: clip albedo to [0, 1] to handle scale ambiguity
        rho = rho.clamp(1e-6, 1.0)

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

pixels_flat, normals_flat, mask = load_render_data("renders")
images_flat = pixels_flat
print(images_flat.shape)
K = pixels_flat.shape[0]
P = normals_flat.shape[0]

rho, L = alternating_least_squares(normals_flat, images_flat, num_iters=500)

# One-time setup
y_tilde = compute_sh_basis_weighted(normals_flat)  # (P, 9)

# Parameters
#L, rho = init_parameters(num_images=K, num_pixels=P)

# Forward pass (this is what you'll wrap in your optimization loop)
predicted = predict_images(rho, L, y_tilde)        # (K, P, 3)
loss = ((predicted - images_flat) ** 2).mean()
print(loss)
save_predicted_images(predicted, mask, output_dir=".")
save_albedo(rho, mask)