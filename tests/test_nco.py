import pytest
from pathlib import Path
from romsviz import nco
import netCDF4 as nc
import numpy as np


FORCING_1 = Path(__file__).parent / 'forcing_1.nc'
FORCING_2 = Path(__file__).parent / 'forcing_2.nc'
FORCING_STAR = str(Path(__file__).parent / 'forcing_*.nc')


@pytest.fixture(scope='module')
def forcing1():
    with nc.Dataset(FORCING_1) as dset:
        dset.set_auto_maskandscale(False)
        dset.set_auto_chartostring(False)
        yield dset


@pytest.fixture(scope='function')
def dset_out():
    import uuid
    fname = uuid.uuid4()
    with nc.Dataset(fname, 'w', diskless=True) as dset:
        yield dset


class Test_horz_slice:
    @pytest.fixture(scope='class')
    def result(self, forcing1):
        import uuid
        fname = uuid.uuid4()
        with nc.Dataset(fname, 'w', diskless=True) as dset_out:
            nco.horz_slice(forcing1, dset_out, depths=[0, 1, 10])
            yield dset_out

    def test_copies_dataset_attributes(self, forcing1, result):
        assert result.getncattr('Conventions') == forcing1.getncattr('Conventions')

    def test_copies_only_nondepth_dimensions(self, result):
        assert 's_rho' not in result.dimensions
        assert 's_w' not in result.dimensions

    def test_copies_only_rho_dimensions(self, result):
        assert 'xi_u' not in result.dimensions
        assert 'xi_v' not in result.dimensions
        assert 'eta_u' not in result.dimensions
        assert 'eta_v' not in result.dimensions

    def test_creates_variables(self, result):
        assert 'zeta' in result.variables
        assert 'u' in result.variables
        assert 'h' in result.variables

    def test_copies_variable_attributes(self, forcing1, result):
        assert result.variables['h'].units == forcing1.variables['h'].units

    def test_copies_nondepth_rho_variables(self, forcing1, result):
        assert 'zeta' in result.variables

        out_var = result.variables['zeta']
        in_var = forcing1.variables['zeta']
        assert out_var[:].tolist() == in_var[:].tolist()

    def test_depth_rho_variables_has_correct_shape(self, forcing1, result):
        assert 'temp' in forcing1.variables
        assert 'temp' in result.variables

        dims_in = forcing1.variables['temp'].dimensions
        assert dims_in == ('ocean_time', 's_rho', 'eta_rho', 'xi_rho')

        dims_out = result.variables['temp'].dimensions
        assert dims_out == ('ocean_time', 'depth', 'eta_rho', 'xi_rho')

        shp_in = forcing1.variables['temp'].shape
        shp_out = result.variables['temp'].shape
        assert shp_out == (shp_in[0], 3, shp_out[2], shp_out[3])

    def test_depth_u_variables_has_correct_shape(self, forcing1, result):
        varname = 'u'
        assert varname in forcing1.variables
        assert varname in result.variables

        dims_in = forcing1.variables[varname].dimensions
        assert dims_in == ('ocean_time', 's_rho', 'eta_u', 'xi_u')

        dims_out = result.variables[varname].dimensions
        assert dims_out == ('ocean_time', 'depth', 'eta_rho', 'xi_rho')

        shp_in = forcing1.variables[varname].shape
        shp_out = result.variables[varname].shape
        assert shp_out == (shp_in[0], 3, shp_out[2], shp_out[3])
