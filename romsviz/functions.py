def open_roms(paths):
    import xarray as xr
    import glob

    if isinstance(paths, str):
        paths = sorted(glob.glob(paths))

    return xr.open_mfdataset(
        paths=paths,
        chunks={'ocean_time': 1},
        concat_dim='ocean_time',
        compat='override',
        coords='minimal',
        data_vars='minimal',
        combine='nested',
        parallel=True,
        join='override',
        combine_attrs='override',
    )


def add_zrho(dset):
    if dset.Vtransform == 1:
        z_rho = dset.hc * (dset.s_rho - dset.Cs_r) + dset.Cs_r * dset.h
    elif dset.Vtransform == 2:
        z_rho = dset.h * (dset.hc * dset.s_rho + dset.Cs_r * dset.h) / (dset.hc + dset.h)
    else:
        raise ValueError(f"Unknown Vtransform: {dset.Vtransform}")

    return dset.assign(z_rho=z_rho.transpose('s_rho', 'eta_rho', 'xi_rho'))


def add_zw(dset):
    if dset.Vtransform == 1:
        z_w = dset.hc * (dset.s_w - dset.Cs_w) + dset.Cs_w * dset.h
    elif dset.Vtransform == 2:
        z_w = dset.h * (dset.hc * dset.s_w + dset.Cs_w * dset.h) / (dset.hc + dset.h)
    else:
        raise ValueError(f"Unknown Vtransform: {dset.Vtransform}")

    return dset.assign(z_rho=z_w.transpose('s_w', 'eta_rho', 'xi_rho'))


def horz_slice(dset, depths):
    import numpy as np
    import xarray as xr

    s_dim = 's_rho'

    if s_dim == 's_rho':
        if 'z_rho' not in dset:
            dset = add_zrho(dset)
        var_z = dset.z_rho
    else:
        if 'z_w' not in dset:
            dset = add_zw(dset)
        var_z = dset.z_w

    var_depths = -xr.Variable('depth', depths)
    kmax = dset.dims[s_dim]  # Number of vertical levels

    # Get layer number above desired depth
    k_above = (var_z < var_depths).sum(dim=s_dim)
    k_above = np.maximum(np.minimum(k_above, kmax - 1), 1)
    dim_order = list(var_depths.dims) + [d for d in var_z.dims if d != s_dim]
    k_above = k_above.transpose(*dim_order)

    # Select layers below and above
    dset_0 = dset.isel({s_dim: k_above - 1})
    dset_1 = dset.isel({s_dim: k_above})

    # Find out where exactly between the layers we should be
    depth_0 = var_z.isel({s_dim: k_above - 1})
    depth_1 = var_z.isel({s_dim: k_above})
    frac = (depth_1 - var_depths) / (depth_1 - depth_0)
    frac = np.minimum(1, np.maximum(0, frac))

    # Use interpolation between layers, for depth-dependent parameters
    depth_vars = {k: None for k, v in dset.variables.items() if s_dim in v.dims}
    for varname in depth_vars:
        v = (1 - frac) * dset_1[varname] + frac * dset_0[varname]
        depth_vars[varname] = v.transpose(*dset_1[varname].dims)
    return dset_1.assign(**depth_vars)


def point(dset, lat, lon):
    pass


def bilin_inv(f, g, F, G, maxiter=7, tol=1.0e-7):
    import numpy as np

    """Inverse bilinear interpolation

    f, g : scalars or arrays of same shape
    F, G : 2D arrays of the same shape

    returns x, y : shaped like f and g
    such that F and G linearly interpolated to x, y
    returns f and g

    """
    imax, jmax = F.shape

    # initial guess
    x = np.zeros_like(f) + 0.5 * imax
    y = np.zeros_like(f) + 0.5 * jmax

    for t in range(maxiter):
        i = x.astype("i4").clip(0, imax - 2)
        j = y.astype("i4").clip(0, jmax - 2)
        p = x - i
        q = y - j

        # Bilinear estimate of F[x,y] and G[x,y]
        Fs = (
            (1 - p) * (1 - q) * F[i, j]
            + p * (1 - q) * F[i + 1, j]
            + (1 - p) * q * F[i, j + 1]
            + p * q * F[i + 1, j + 1]
        )
        Gs = (
            (1 - p) * (1 - q) * G[i, j]
            + p * (1 - q) * G[i + 1, j]
            + (1 - p) * q * G[i, j + 1]
            + p * q * G[i + 1, j + 1]
        )

        H = (Fs - f) ** 2 + (Gs - g) ** 2

        if np.all(H < tol):
            break

        # Estimate Jacobi matrix
        Fx = (1 - q) * (F[i + 1, j] - F[i, j]) + q * (F[i + 1, j + 1] - F[i, j + 1])
        Fy = (1 - p) * (F[i, j + 1] - F[i, j]) + p * (F[i + 1, j + 1] - F[i + 1, j])
        Gx = (1 - q) * (G[i + 1, j] - G[i, j]) + q * (G[i + 1, j + 1] - G[i, j + 1])
        Gy = (1 - p) * (G[i, j + 1] - G[i, j]) + p * (G[i + 1, j + 1] - G[i + 1, j])

        # Newton-Raphson step
        # Jinv = np.linalg.inv([[Fx, Fy], [Gx, Gy]])
        # incr = - np.dot(Jinv, [Fs-f, Gs-g])
        # x = x + incr[0], y = y + incr[1]
        det = Fx * Gy - Fy * Gx
        x -= (Gy * (Fs - f) - Fy * (Gs - g)) / det
        y -= (-Gx * (Fs - f) + Fx * (Gs - g)) / det

    return x, y
