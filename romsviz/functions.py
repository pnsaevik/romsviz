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
    var_z = dset.z_rho if s_dim == 's_rho' else dset.z_w

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
