from pathlib import Path
from romsviz import functions


FORCING_1 = Path(__file__).parent / 'forcing_1.nc'
FORCING_2 = Path(__file__).parent / 'forcing_2.nc'
FORCING_STAR = str(Path(__file__).parent / 'forcing_*.nc')


class Test_open_roms:
    def test_can_interpret_glob_patterns(self):
        with functions.open_roms(FORCING_STAR) as dset:
            assert dset.dims['ocean_time'] == 5

    def test_timeless_variables_are_not_combined(self):
        with functions.open_roms(FORCING_STAR) as dset:
            assert dset.h.dims == ('eta_rho', 'xi_rho')
            assert dset.s_rho.dims == ('s_rho', )
