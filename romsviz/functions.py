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
