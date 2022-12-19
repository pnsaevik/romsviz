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
    with xr.open_dataset(FORCING_1, decode_cf=False, mask_and_scale=False) as dset:
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
        assert dset.z_w.dims == ('s_w', 'eta_rho', 'xi_rho')


class Test_horz_slice:
    def test_temp_has_correct_dimensions(self, forcing1):
        keep_vars = ['temp', 'z_rho', 's_rho', 'Vtransform', 'Cs_r', 'h', 'hc']
        dset = forcing1.drop_vars([v for v in forcing1.variables if v not in keep_vars])
        hslice = functions.horz_slice(dset, depths=[0, 10, 20])
        assert hslice.temp.dims == ('ocean_time', 'depth', 'eta_rho', 'xi_rho')

    def test_w_has_correct_dimensions(self, forcing1):
        keep_vars = ['w', 'z_w', 's_w', 'Vtransform', 'Cs_w', 'h', 'hc']
        dset = forcing1.drop_vars([v for v in forcing1.variables if v not in keep_vars])
        hslice = functions.horz_slice(dset, depths=[0, 10, 20])
        assert hslice.w.dims == ('ocean_time', 'depth', 'eta_rho', 'xi_rho')

    def test_zrho_is_correct(self, forcing1):
        hslice = functions.horz_slice(forcing1, depths=[3])
        zrho_values = hslice.z_rho.values
        assert np.all(np.isclose(zrho_values, -3))

    def test_zw_is_correct(self, forcing1):
        hslice = functions.horz_slice(forcing1, depths=[3])
        zw_values = hslice.z_w.values
        assert np.all(np.isclose(zw_values, -3))


class Test_point:
    def test_returns_single_point(self, forcing1):
        point = functions.point(forcing1, lat=59.03062209, lon=5.67321047)
        assert point.temp.dims == ('ocean_time', 's_rho')

    def test_can_add_depths_afterwards(self, forcing1):
        point = functions.point(forcing1, lat=59.03062209, lon=5.67321047)
        point = functions.add_zw(point)
        point = functions.add_zrho(point)
        assert len(point.z_rho) == 35
        assert len(point.z_w) == 36


class Test_cell:
    def test_returns_single_point(self, forcing1):
        point = functions.cell(forcing1, lat=59.03062209, lon=5.67321047)
        assert point.temp.dims == ('ocean_time', 's_rho')

    def test_can_add_depths_afterwards(self, forcing1):
        point = functions.cell(forcing1, lat=59.03062209, lon=5.67321047)
        point = functions.add_zw(point)
        point = functions.add_zrho(point)
        assert len(point.z_rho) == 35
        assert len(point.z_w) == 36


class Test_velocity:
    def test_returns_absolute_velocity_if_no_params(self, forcing1):
        velocity = functions.velocity(forcing1)
        assert np.all(velocity.values >= 0)

    def test_shrinks_domain_size(self, forcing1):
        velocity = functions.velocity(forcing1)
        assert velocity.shape[-2:] == (8, 13)
        assert forcing1.h.shape == (10, 15)

    def test_returns_zero_velocity_if_on_land(self, forcing1):
        x = 2
        y = 4
        velocity = functions.velocity(forcing1)
        assert forcing1.mask_rho[y, x].values == 0
        assert velocity[0, 0, y - 1, x - 1].values == 0

    def test_can_handle_point_datasets(self, forcing1):
        point = functions.point(forcing1, lat=59.03062209, lon=5.67321047)
        velocity = functions.velocity(point)
        assert velocity.dims == ('ocean_time', 's_rho')

    def test_can_handle_azimuthal_point_datasets(self, forcing1):
        point = functions.point(forcing1, lat=59.03062209, lon=5.67321047)
        velocity = functions.velocity(point, azimuth=0)
        assert velocity.dims == ('ocean_time', 's_rho')

    def test_returns_zero_velocity_if_on_land_if_point_dataset(self, forcing1):
        point = functions.point(forcing1, lat=59.026137, lon=5.672106)
        assert point.mask_rho < 0.001
        velocity = functions.velocity(point)
        assert np.all(velocity.values < 0.001)

    def test_velocity_changes_with_azimuth(self, forcing1):
        velocity_north = functions.velocity(forcing1, azimuth=0)
        velocity_east = functions.velocity(forcing1, azimuth=-np.pi/2)
        assert velocity_north.dims == ('ocean_time', 's_rho', 'eta_rho', 'xi_rho')
        assert velocity_north[0, 0, 0, 0] != velocity_east[0, 0, 0, 0]


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


class Test_select_layer:
    def test_equals_regular_selector(self):
        import dask.array
        data = np.arange(96).reshape((4, 2, 3, 4))
        variable = xr.Variable(
            dims=('ocean_time', 's_rho', 'eta_rho', 'xi_rho'),
            data=dask.array.from_array(data, chunks=(1, None, None, None)),
        )
        srho_selector = xr.Variable(
            dims=('depth', 'eta_rho', 'xi_rho'),
            data=[
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 0]],
                [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
            ],
        )
        selector = {'s_rho': srho_selector}
        result = functions.select_layer(variable, selector).compute()
        assert result[0].values.tolist() == [
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
            [[0, 1, 14, 15], [4, 17, 18, 19], [20, 21, 22, 11]],
            [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]],
        ]
        assert result[3].values.tolist() == [
            [[72, 73, 74, 75], [76, 77, 78, 79], [80, 81, 82, 83]],
            [[72, 73, 86, 87], [76, 89, 90, 91], [92, 93, 94, 83]],
            [[84, 85, 86, 87], [88, 89, 90, 91], [92, 93, 94, 95]],
        ]
