import pytest
from pathlib import Path
from romsviz import functions
import xarray as xr


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
