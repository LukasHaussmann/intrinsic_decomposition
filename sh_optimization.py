import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def random_sh_lighting(num_images=4, seed=32):
    """
    Generate random SH coefficients with realistic frequency falloff.
    Returns (K, 3, 9).
    """
    rng = np.random.default_rng(seed)

    K = num_images
    L = np.zeros((K, 9))

    # Band-wise std: lower bands are dominant
    band_scale = np.array([
        1.0,                      # l=0 (DC)
        0.5, 0.5, 0.5,            # l=1
        0.25, 0.25, 0.25, 0.25, 0.25  # l=2
    ])

    for k in range(K):
        # Sample a base "color temperature" for this image
        # so different images have visibly different lighting
        color_bias = 0.7 + 0.6 * rng.random(1)   # per-channel scale, [0.7, 1.3]

        # Gaussian samples scaled by band falloff
        coeffs = rng.random(9) * band_scale

        # Force the DC term to be positive and dominant
        # so the lighting is mostly positive everywhere
        coeffs[0] = (1.5 + 0.5 * rng.random(1).item()) * color_bias

        L[k] = coeffs

    return L
def sh_basis_weighted(normals):
    x = normals[..., 0]
    y = normals[..., 1]
    z = normals[..., 2]
    pi = math.pi
    c0 = 0.5 * math.sqrt(pi)              # band 0
    c1 = (pi / 3.0) * math.sqrt(3.0 / pi) # band 1
    c2 = (pi / 8.0) * math.sqrt(15.0 / pi) # band 2, m = ±2 off-diag and m = ±1
    c3 = (pi / 16.0) * math.sqrt(5.0 / pi) # band 2, m = 0
    c4 = (pi / 16.0) * math.sqrt(15.0 / pi) # band 2, m = +2

    y_tilde = np.stack([
        c0 * np.ones_like(x),
        c1 * y,                   # i=1:  y
        c1 * z,                   # i=2:  z
        c1 * x,                   # i=3:  x
        c2 * x * y,               # i=4:  xy
        c2 * y * z,               # i=5:  yz
        c3 * (3.0 * z * z - 1.0), # i=6:  3z² - 1
        c2 * x * z,               # i=7:  xz
        c4 * (x * x - y * y),     # i=8:  x² - y²
    ], axis=-1)
    return y_tilde

def plot_metrics(losses, maes, sh_maes, save_path="sh_optimization_metrics.png"):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
    for ax, values, ylabel, title in [
        (ax1, losses,  "MSE", "Reconstruction Loss"),
        (ax2, maes,    "MAE", "Albedo MAE (aligned)"),
        (ax3, sh_maes, "MAE", "SH Params MAE (aligned)"),
    ]:
        ax.plot(values)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.grid(True, which="major", linestyle="-",  linewidth=0.6, alpha=0.7)
        ax.grid(True, which="minor", linestyle="--", linewidth=0.3, alpha=0.4)
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        ax.tick_params(axis="both", which="both", direction="in")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved metrics plot to {save_path}")
    plt.close(fig)

"""
Simplest case:
    - pixels: take p values for albedo
    - lighting: n vectors of sh coefficients
    - normals: p random vectors
    -> rendering: create n renderings one for each set of sh coefficients
"""
rng = np.random.default_rng(42)
p = 100
n = 2
albedo = rng.random((p,1))
normals = rng.random((p,3))
normals /= np.linalg.norm(normals, axis=1, keepdims=True)
sh_params = random_sh_lighting(num_images=n)
print(sh_params.shape)
sh_basis = sh_basis_weighted(normals)
print(sh_basis.shape)
renderings = albedo.T * sh_params.dot(sh_basis.T)
print(renderings.shape)

rho_estimate = np.ones((p,1))
sh_estimate = np.zeros((n,9))
for i in range(n):
    sh_estimate[i,0] = 1.5

num_steps = 100_000
use_als = False
use_fast_als = True
if not use_als:
    L = torch.nn.Parameter(torch.from_numpy(sh_estimate))
    rho = torch.nn.Parameter(torch.from_numpy(rho_estimate))
    y_tilde = torch.tensor(torch.from_numpy(sh_basis))
    images = torch.tensor(torch.from_numpy(renderings))
    optimizer = torch.optim.Adam([rho, L], lr=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1e-5)

    losses, maes, sh_maes = [], [], []
    for step in range(num_steps):
        optimizer.zero_grad()
        pred = (rho * y_tilde @ L.T).T
        loss = ((images - pred)**2).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            scale = rho.mean()
            rho.div_(scale)
            L.mul_(scale)
        print(step, ':', loss.item())
        rho_np = rho.detach().numpy()
        alpha = albedo.T.dot(rho_np) / rho_np.T.dot(rho_np)
        mae_gd = np.mean(np.abs(albedo - alpha * rho_np))
        sh_mae = np.mean(np.abs(L.detach().numpy() / alpha - sh_params))
        losses.append(loss.item())
        maes.append(float(mae_gd))
        sh_maes.append(float(sh_mae))
        #print(step,'mae gd',mae_gd)
        #print(sh_estimate / alpha)

    rho_np = rho.detach().numpy()
    alpha = albedo.T.dot(rho_np) / rho_np.T.dot(rho_np)
    mae_gd = np.mean(np.abs(albedo - alpha * rho_np))
    print('mae gd',mae_gd)
    print(sh_estimate / alpha)
    print(sh_params)
    #print(rho_np * alpha)
    #print(albedo)
    plot_metrics(losses, maes, sh_maes, 'sh_optimization_gradient_descent_metrics.png')
elif use_fast_als:

    losses, maes, sh_maes = [], [], []
    for step in range(num_steps):
        y_t = rho_estimate * sh_basis          # (P, 9)
        A = y_t.T @ y_t                        # (9, 9)
        b = y_t.T @ renderings.T              # (9, n)
        sh_estimate = np.linalg.solve(A, b).T  # (n, 9)

        shading = sh_estimate.dot(sh_basis.T)
        numerator = np.sum(renderings * shading, axis=0)
        denominator = np.sum(shading**2, axis=0)
        rho_estimate = (numerator / (denominator + 1e-12))[:,None]
        scale = np.mean(rho_estimate)
        rho_estimate /= scale
        sh_estimate *= scale

        reproduced = rho_estimate.T * sh_estimate.dot(sh_basis.T)
        loss = np.sum(np.abs(reproduced - renderings))
        print(step, loss)
        alpha = albedo.T.dot(rho_estimate) / rho_estimate.T.dot(rho_estimate)
        mae_als = np.mean(np.abs(albedo - alpha * rho_estimate))
        sh_mae = np.mean(np.abs(sh_estimate / alpha - sh_params))
        losses.append(float(loss))
        maes.append(float(mae_als))
        sh_maes.append(float(sh_mae))

    alpha = albedo.T.dot(rho_estimate) / rho_estimate.T.dot(rho_estimate)
    print('mae fast als', np.mean(np.abs(albedo - alpha * rho_estimate)))
    print(sh_estimate / alpha)
    print(sh_params)
    plot_metrics(losses, maes, sh_maes, 'sh_optimization_fast_als_metrics.png')
else:

    losses, maes, sh_maes = [], [], []
    for step in range(num_steps):
        # keep albedo const and solve the linear system per image for sh params
        for i in range(n):
            # multiply albedo into sh basis
            y_t = rho_estimate * sh_basis
            # y_t: (P,9)
            # sh_estimate: (N,9) -> sh_estimate[i]: (9,)
            # renderings: (N,P) -> renderings[i]: (P,)
            # solve renderings[i] = y_t @ sh_estimate[i]
            # -> sh_estimate[i]* = (y_t.T @ y_t).inv @ y_t.T @ renderings[i]
            # (y_t.T @ y_t).inv : (9,9)
            # (y_t.T @ y_t).inv @ y_t.T : (9,P)
            # (y_t.T @ y_t).inv @ y_t.T @ renderings[i] : (9,)
            # SVD: y_t = U Σ V.T
            # pseudoinverse
            U, s, Vt = np.linalg.svd(y_t, full_matrices=False)
            S_pinv = Vt.T @ np.diag(1.0 / s) @ U.T
            sh_estimate[i] = S_pinv @ renderings[i]

        # sh_estimate: (N,9), sh_basis: (P,9)
        shading = sh_estimate.dot(sh_basis.T)
        numerator = np.sum(renderings * shading, axis=0)
        denominator = np.sum(shading**2, axis=0)
        rho_estimate = (numerator / (denominator + 1e-12))[:,None]
        scale = np.mean(rho_estimate)
        rho_estimate /= scale
        sh_estimate *= scale

        reproduced = rho_estimate.T * sh_estimate.dot(sh_basis.T)
        loss = np.sum(np.abs(reproduced - renderings))
        print(step, loss)
        alpha = albedo.T.dot(rho_estimate) / rho_estimate.T.dot(rho_estimate)
        mae_als = np.mean(np.abs(albedo - alpha * rho_estimate))
        sh_mae = np.mean(np.abs(sh_estimate / alpha - sh_params))
        losses.append(loss.item())
        maes.append(float(mae_als))
        sh_maes.append(float(sh_mae))
        #print(step,'mae als',np.mean(np.abs(albedo - alpha * rho_estimate)))
        #print(sh_estimate/alpha)

    # rho_gt = rho_est * alpha
    # for i in p: min(alpha) sum(rho_gt[i] - rho_est[i] * alpha)^2

    alpha = albedo.T.dot(rho_estimate) / rho_estimate.T.dot(rho_estimate)
    print('mae als',np.mean(np.abs(albedo - alpha * rho_estimate)))
    print(sh_estimate / alpha)
    print(sh_params)
    plot_metrics(losses, maes, sh_maes, 'sh_optimization_als_metrics.png')
