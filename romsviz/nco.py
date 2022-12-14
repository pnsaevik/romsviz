"""
The module contains out-of-core functions for nc file manipulation based on the
netCDF4 library
"""

import numpy as np
import logging


logger = logging.getLogger(__name__)


def horz_slice(dset_in, dset_out, depths):
    # Copy dataset attributes
    dset_attrs = {k: dset_in.getncattr(k) for k in dset_in.ncattrs()}
    dset_out.setncatts(dset_attrs)

    # Copy dimensions that are not related to depth, time or u/v points
    skip_dims = ['s_rho', 's_w', 'ocean_time', 'xi_u', 'eta_u', 'xi_v', 'eta_v']
    copydims = [d for d in dset_in.dimensions if d not in skip_dims]
    for dimname in copydims:
        size = dset_in.dimensions[dimname].size
        dset_out.createDimension(dimname, size)

    # Create new dimensions
    dset_out.createDimension('ocean_time', None)  # Let time be unlimited, to allow chunkwise operation
    dset_out.createDimension('depth', len(depths))

    # Create variables
    for varname in dset_in.variables:
        var_in = dset_in.variables[varname]
        dtype = var_in.dtype
        dims_in = var_in.dimensions
        attrs_in = {k: var_in.getncattr(k) for k in var_in.ncattrs()}
        dims_out = replace_items(
            dims_in, s_rho='depth', s_w='depth', xi_u='xi_rho', eta_u='eta_rho',
            xi_v='xi_rho', eta_v='eta_rho',
        )
        if list(dims_out) == ['depth', ]:
            dims_out = ['depth', 'eta_rho', 'xi_rho']
        fill_value = attrs_in.get('_FillValue', False)
        var_out = dset_out.createVariable(
            varname=varname,
            datatype=dtype,
            dimensions=dims_out,
            fill_value=fill_value,
        )
        var_out.setncatts(attrs_in)
        var_out.set_auto_maskandscale(False)
        var_out.set_auto_chartostring(False)

    # Create depth interpolation arrays
    cached_arrays = create_s_arrays(depths, dset_in)

    # Copy variables chunk-wise
    for time_idx in range(dset_in.dimensions['ocean_time'].size):
        time_str = str(dset_in.variables['ocean_time'][time_idx])
        logger.debug(f'Copy time step {time_str}')
        print(f'Copy time step {time_str}')
        for varname in dset_in.variables:
            copy_chunk(dset_in, dset_out, varname, time_idx, cached_arrays)


def copy_chunk(dset_in, dset_out, varname, time_idx, cached_arrays):
    var_in = dset_in[varname]

    # Read data chunk
    first_dim = var_in.dimensions[0] if len(var_in.dimensions) else None
    if first_dim == 'ocean_time':
        data_in = var_in[time_idx]
        dims = var_in.dimensions[1:]
    elif time_idx == 0:
        data_in = var_in[:]
        dims = var_in.dimensions
    else:
        return

    # Set ocean velocities to zero on land
    if varname in ['u', 'ubar', 'uice']:
        if 'mask_u' not in cached_arrays:
            cached_arrays['mask_u'] = dset_in['mask_u'][:]
        data_in *= cached_arrays['mask_u']
    elif varname in ['v', 'vbar', 'vice']:
        if 'mask_v' not in cached_arrays:
            cached_arrays['mask_v'] = dset_in['mask_v'][:]
        data_in *= cached_arrays['mask_v']

    # Interpolate u/v points to rho points
    if 'xi_u' in dims or 'eta_v' in dims:
        dimname = 'xi_u' if 'xi_u' in dims else 'eta_v'
        shp = [dset_in.dimensions[d].size for d in dims]
        dim_idx = next(i for i, d in enumerate(dims) if d == dimname)
        shp[dim_idx] += 1
        data_tmp = np.empty(shp, data_in.dtype)
        idx_start = tuple([0 if d == dimname else slice(None) for d in dims])
        idx_stop = tuple([-1 if d == dimname else slice(None) for d in dims])
        idx_first = tuple([slice(None, -1) if d == dimname else slice(None) for d in dims])
        idx_last = tuple([slice(1, None) if d == dimname else slice(None) for d in dims])
        idx_mid = tuple([slice(1, -1) if d == dimname else slice(None) for d in dims])
        data_tmp[idx_start] = 0
        data_tmp[idx_stop] = 0
        data_tmp[idx_mid] = 0.5 * (data_in[idx_first] + data_in[idx_last])
        data_in = data_tmp

    # Interpolate data in depth
    if 's_w' in dims:
        s_int = cached_arrays['k_w']
        s_frac = cached_arrays['f_w']
        data_out = make_slice(data_in, dims, s_int, s_frac)
    elif 's_rho' in dims:
        s_int = cached_arrays['k_rho']
        s_frac = cached_arrays['f_rho']
        data_out = make_slice(data_in, dims, s_int, s_frac)
    else:
        data_out = data_in

    # Write data chunk
    if first_dim == 'ocean_time':
        dset_out.variables[varname][time_idx] = data_out
    else:
        dset_out.variables[varname][:] = data_out


def replace_items(iterable, **kwargs):
    items = []
    for item in iterable:
        if item in kwargs:
            item = kwargs[item]
        items.append(item)
    return items


def make_slice(data, dims, s_int, s_frac):
    dims = replace_items(
        dims, s_w='s_rho', xi_u='xi_rho', eta_u='eta_rho',
        xi_v='xi_rho', eta_v='eta_rho',
    )
    if len(dims) == 0 or dims[0] != 's_rho':
        return data

    nd, ny, nx = s_int.shape
    if len(dims) == 1:
        data = np.broadcast_to(data[..., np.newaxis, np.newaxis], data.shape + (ny, nx))
    elif list(dims)[1:] != ['eta_rho', 'xi_rho']:
        raise NotImplementedError

    idx_y, idx_x = np.meshgrid(range(ny), range(nx), indexing='ij')
    i = np.broadcast_to(idx_x, (nd, ny, nx)).ravel()
    j = np.broadcast_to(idx_y, (nd, ny, nx)).ravel()
    k = s_int.ravel()
    f = s_frac.ravel()

    data_0 = data[k, j, i]
    data_1 = data[k + 1, j, i]
    result = (1 - f) * data_0 + f * data_1

    return result.reshape((nd, ny, nx))


def create_s_arrays(depths, dset):
    z_rho = create_z(dset, 'rho')
    z_w = create_z(dset, 'w')

    k_rho, f_rho = create_s_array(depths, z_rho)
    k_w, f_w = create_s_array(depths, z_w)

    return dict(
        k_w=k_w,
        f_w=f_w,
        k_rho=k_rho,
        f_rho=f_rho,
    )


def create_s_array(depths, z):
    ns, ny, nx = z.shape
    nd = len(depths)
    y, x = np.meshgrid(range(ny), range(nx), indexing='ij')
    i = np.broadcast_to(x, (nd, ny, nx)).ravel()
    j = np.broadcast_to(y, (nd, ny, nx)).ravel()
    d = np.broadcast_to(np.reshape(depths, (-1, 1, 1)), (nd, ny, nx)).ravel()
    k = np.sum(z < -np.asarray(depths).reshape((-1, 1, 1, 1)), axis=1).ravel()
    k = np.maximum(1, np.minimum(ns - 1, k))
    z_0 = z[k - 1, j, i]
    z_1 = z[k, j, i]
    f = (d + z_0) / (z_0 - z_1)
    return (k - 1).reshape((nd, ny, nx)), f.reshape((nd, ny, nx))


def create_z(dset, stagger):
    s, cs = get_s_cs(dset, stagger)
    vtrans = dset.variables['Vtransform'][:]
    h = dset.variables['h'][:]
    hc = dset.variables['hc'][:]

    if vtrans == 2:
        z = h * (hc * s + cs * h) / (hc + h)
    else:
        raise ValueError(f"Not implemented: Vtransform = {vtrans}")

    return z


def get_s_cs(dset, stagger):
    if stagger == 'rho':
        num_levels = dset.dimensions['s_rho'].size
        s = np.linspace(-1, 0, 1 + 2 * num_levels)[1::2][:, np.newaxis, np.newaxis]
        cs = dset.variables['Cs_r'][:][:, np.newaxis, np.newaxis]
    elif stagger == 'w':
        num_levels = dset.dimensions['s_w'].size
        s = np.linspace(-1, 0, num_levels)[:, np.newaxis, np.newaxis]
        cs = dset.variables['Cs_w'][:][:, np.newaxis, np.newaxis]
    else:
        raise ValueError(f"Unknown: stagger = {stagger}")

    return s, cs


def _copy_dataset_attributes(dset_in, dset_out):
    dset_attrs = {k: dset_in.getncattr(k) for k in dset_in.ncattrs()}
    dset_out.setncatts(dset_attrs)
