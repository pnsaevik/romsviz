import pytest
from pathlib import Path
from romsviz import functions
import xarray as xr
import numpy as np


FORCING_1 = Path(__file__).parent / 'forcing_1.nc'
FORCING_2 = Path(__file__).parent / 'forcing_2.nc'
FORCING_STAR = str(Path(__file__).parent / 'forcing_*.nc')


@pytest.fixture(scope='module')
def forcing1():
    with xr.open_dataset(FORCING_1) as dset:
        yield dset


class Test_open_roms:
    def test_can_interpret_glob_patterns(self):
        with functions.open_roms(FORCING_STAR) as dset:
            assert dset.dims['ocean_time'] == 5

    def test_timeless_variables_are_not_combined(self):
        with functions.open_roms(FORCING_STAR) as dset:
            assert dset.h.dims == ('eta_rho', 'xi_rho')
            assert dset.s_rho.dims == ('s_rho', )


class Test_add_zrho:
    def test_zrho_has_correct_dimensions(self, forcing1):
        dset = functions.add_zrho(forcing1)
        assert dset.z_rho.dims == ('s_rho', 'eta_rho', 'xi_rho')


class Test_add_zw:
    def test_zw_has_correct_dimensions(self, forcing1):
        dset = functions.add_zw(forcing1)
        assert dset.z_rho.dims == ('s_w', 'eta_rho', 'xi_rho')


class Test_horz_slice:
    def test_temp_has_correct_dimensions(self, forcing1):
        dset = functions.add_zrho(forcing1)
        keep_vars = ['temp', 'z_rho', 's_rho']
        subset = dset.drop_vars([v for v in dset.variables if v not in keep_vars])
        hslice = functions.horz_slice(subset, depths=[0, 10, 20])
        assert hslice.temp.dims == ('ocean_time', 'depth', 'eta_rho', 'xi_rho')

    def test_zrho_is_correct(self, forcing1):
        dset = functions.add_zrho(forcing1)
        hslice = functions.horz_slice(dset, depths=[3])
        zrho_values = hslice.z_rho.values
        assert np.all(np.isclose(zrho_values, -3))


class Test_point:
    def test_includes_depth(self, forcing1):
        pass


class Test_bilin_inv:
    def test_can_retrieve_coordinates(self):
        y, x = np.meshgrid(np.arange(4), np.arange(5), indexing='ij')
        u = x + y
        v = x - y
        i = np.array([0, 1, 4])
        j = np.array([0, 0, 3])
        u_ji = u[j, i]
        v_ji = v[j, i]

        j2, i2 = functions.bilin_inv(u_ji, v_ji, u, v)
        assert j2.tolist() == j.tolist()
        assert i2.tolist() == i.tolist()
