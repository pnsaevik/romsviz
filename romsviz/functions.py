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
