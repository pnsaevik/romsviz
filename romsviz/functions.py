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
        decode_cf=False,
        mask_and_scale=False,
    )


def add_zrho(dset):
    import numpy as np
    import xarray as xr
    s_rho_values = np.linspace(-1, 0, 1 + 2 * dset.dims['s_rho'])[1::2]
    s_rho = xr.Variable('s_rho', s_rho_values)

    if dset.Vtransform == 1:
        z_rho = dset.hc * (s_rho - dset.Cs_r) + dset.Cs_r * dset.h
    elif dset.Vtransform == 2:
        z_rho = dset.h * (dset.hc * s_rho + dset.Cs_r * dset.h) / (dset.hc + dset.h)
    else:
        raise ValueError(f"Unknown Vtransform: {dset.Vtransform}")

    dims = tuple(s_rho.dims) + tuple(dset.h.dims)
    return dset.assign(z_rho=z_rho.transpose(*dims))


def add_zw(dset):
    import numpy as np
    import xarray as xr
    s_w_values = np.linspace(-1, 0, dset.dims['s_w'])
    s_w = xr.Variable('s_w', s_w_values)

    if dset.Vtransform == 1:
        z_w = dset.hc * (s_w - dset.Cs_w) + dset.Cs_w * dset.h
    elif dset.Vtransform == 2:
        z_w = dset.h * (dset.hc * s_w + dset.Cs_w * dset.h) / (dset.hc + dset.h)
    else:
        raise ValueError(f"Unknown Vtransform: {dset.Vtransform}")

    dims = tuple(s_w.dims) + tuple(dset.h.dims)
    return dset.assign(z_w=z_w.transpose(*dims))


def horz_slice(dset, depths):
    # Interpolate in s_w direction if necessary
    if 's_w' in dset.dims:
        dset = horz_slice_single_stagger(dset, depths, s_dim='s_w')

    # Interpolate in s_rho direction if necessary
    if 's_rho' in dset.dims:
        dset = horz_slice_single_stagger(dset, depths, s_dim='s_rho')

    return dset


def horz_slice_single_stagger(dset, depths, s_dim='s_rho'):
    import numpy as np
    import xarray as xr

    if s_dim == 's_rho':
        if 'z_rho' not in dset:
            dset = add_zrho(dset)
            dset['z_rho'] = dset.z_rho.compute()
        var_z = dset.z_rho.variable
    else:
        if 'z_w' not in dset:
            dset = add_zw(dset)
            dset['z_w'] = dset.z_w.compute()
        var_z = dset.z_w.variable

    var_depths = -xr.Variable('depth', depths)
    kmax = dset.dims[s_dim]  # Number of vertical levels

    # Get layer number above desired depth
    k_above = (var_z < var_depths).sum(dim=s_dim)
    k_above = np.maximum(np.minimum(k_above, kmax - 1), 1)
    dim_order = list(var_depths.dims) + [d for d in var_z.dims if d != s_dim]
    k_above = k_above.transpose(*dim_order).compute()

    # Find out where exactly between the layers we should be
    depth_0 = var_z.isel({s_dim: k_above - 1})
    depth_1 = var_z.isel({s_dim: k_above})
    frac = (depth_1 - var_depths) / (depth_1 - depth_0)
    frac = np.minimum(1, np.maximum(0, frac))

    # Use interpolation between layers, for depth-dependent parameters
    depth_vars = {
        k: None for k, v in dset.variables.items()
        if v.dims[-3:] == (s_dim, 'eta_rho', 'xi_rho')
    }
    for varname in depth_vars:
        # Interpolate between layers above and below
        var_0 = select_layer(dset.variables[varname], {s_dim: k_above - 1})
        var_1 = select_layer(dset.variables[varname], {s_dim: k_above})
        v = (1 - frac) * var_1 + frac * var_0
        depth_vars[varname] = v.transpose(*var_0.dims)
    return dset.assign(**depth_vars)


def select_layer(variable, selector):
    """This function is requried since .isel is not properly lazified"""

    # TODO: This does not work for u-points and v-points

    import xarray as xr
    import dask.array

    s_dim = next(s for s in selector)
    z_dim = selector[s_dim].dims[0]

    # Compute shape and dimensions of output object
    variable_dims = {k: v for k, v in zip(variable.dims, variable.shape)}
    selector_dims = {k: v for k, v in zip(selector[s_dim].dims, selector[s_dim].shape)}
    chunk_dim = 'ocean_time'
    low_dims = list(selector[s_dim].dims[1:])
    low_shape = list(selector[s_dim].shape[1:])
    high_dims = [d for d in variable.dims if d not in [z_dim, s_dim] + low_dims]
    high_shape = [variable_dims[d] for d in high_dims]
    out_dims = high_dims + [z_dim] + low_dims
    out_shape = high_shape + [selector_dims[z_dim]] + low_shape
    chunk_size = [1 if d == chunk_dim else None for d in out_dims]

    class DaskCompatibleObject:
        def __init__(self):
            self.shape = out_shape
            self.ndim = len(out_shape)
            self.dtype = variable.dtype

        def __getitem__(self, item):
            if variable.dims[0] == chunk_dim:
                first_index = item[0]
                new_item = (slice(0, 1),) + item[1:]
                return variable[first_index].compute().isel(selector).values[new_item]
            else:
                return variable.compute().isel(selector).values[item]

    data = dask.array.from_array(DaskCompatibleObject(), chunks=chunk_size, asarray=False)

    return xr.Variable(
        dims=out_dims,
        data=data,
        attrs=variable.attrs,
        encoding=variable.encoding,
    )


def point(dset, lat, lon):
    import numpy as np

    y, x = bilin_inv(lat, lon, dset.lat_rho.values, dset.lon_rho.values)

    x_min = 0.5
    y_min = 0.5
    x_max = dset.dims['xi_rho'] - 1.5
    y_max = dset.dims['eta_rho'] - 1.5
    x = np.clip(x, x_min, x_max)
    y = np.clip(y, y_min, y_max)

    if 'u' in dset:
        dset = dset.assign(u=dset.u * dset.mask_u)
    if 'v' in dset:
        dset = dset.assign(v=dset.v * dset.mask_v)

    coords = dict(
        xi_rho=x,
        eta_rho=y,
        xi_u=x - 0.5,
        eta_u=int(y + 0.5),
        xi_v=int(x + 0.5),
        eta_v=y - 0.5,
    )

    # Do interpolation
    dset = dset.drop_vars(names=list(coords), errors='ignore')
    dset = dset.interp(coords)

    return dset


def velocity(dset, azimuth=None):
    import numpy as np
    import xarray as xr

    u = dset.u.variable
    v = dset.v.variable

    if 'xi_u' in u.dims:
        u = u * dset.mask_u.variable
        v = v * dset.mask_v.variable
        dset = dset.isel(xi_rho=slice(1, -1), eta_rho=slice(1, -1))
        u = 0.5 * (u[..., 1:-1, :-1] + u[..., 1:-1, 1:])
        v = 0.5 * (v[..., :-1, 1:-1] + v[..., 1:, 1:-1])
        u = xr.DataArray(u).rename(xi_u='xi_rho', eta_u='eta_rho')
        v = xr.DataArray(v).rename(xi_v='xi_rho', eta_v='eta_rho')

    if azimuth is None:
        return np.sqrt(u * u + v * v)

    else:
        angle = dset.angle.variable
        assert angle.attrs['units'] == "radians"

        theta = azimuth + np.pi / 2 - angle
        return u * np.cos(theta) + v * np.sin(theta)


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
