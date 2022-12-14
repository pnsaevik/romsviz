def run(*argv):
    subcommands = [
        slice,
        average,
    ]

    parser = get_argument_parser(subcommands)
    args = parser.parse_args(argv)

    import inspect
    func = next(fn for fn in subcommands if fn.__name__ == args.subcmd)
    func_args = {k: getattr(args, k) for k in inspect.getfullargspec(func).args}
    func(**func_args)


def get_argument_parser(func_list):
    import argparse
    from . import __version__ as version_str

    prog_name = __name__.split('.')[0]

    parser = argparse.ArgumentParser(
        prog=prog_name,
        description=(
            "Tools for ROMS plotting and data extraction"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--version', action='version', version=f"{prog_name} {version_str}")
    parser.add_argument('--verbose', '-v', action='count', default=0,
                        help="increase verbosity")

    subparsers = parser.add_subparsers(
        description='Run `romsviz (subcmd) --help` for more information about each subcommand',
        metavar='subcmd',
        dest='subcmd',
    )

    for subfn in func_list:
        subcmd = subfn.__name__
        subparser = subparsers.add_parser(
            name=subcmd,
            help=first_line_of_docstring(subfn),
            description=docstring_description(subfn),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        for arg_name, arg_help in argument_docstring(subfn).items():
            subparser.add_argument(arg_name, help=arg_help)

    parser.romsviz_subparsers = subparsers

    return parser


def first_line_of_docstring(fn):
    import inspect
    if fn.__doc__ is None:
        txt = ""
    else:
        txt = inspect.cleandoc(fn.__doc__)
    lines = txt.split('\n')
    if len(lines) > 0:
        return lines[0]
    else:
        return ""


def docstring_description(fn):
    import inspect
    if fn.__doc__ is None:
        txt = ""
    else:
        txt = inspect.cleandoc(fn.__doc__)
    txt = txt[:txt.find(':param')]
    return txt.strip()


def argument_docstring(fn):
    import inspect
    import re
    args = inspect.getfullargspec(fn).args
    docstrings = {arg: "" for arg in args}

    if fn.__doc__ is None:
        docstring = ""
    else:
        docstring = inspect.cleandoc(fn.__doc__)

    doclines = re.findall(r":param\s+(.*?):\s+([^:]*)", docstring, flags=re.DOTALL)
    for docline in doclines:
        varname = docline[0]
        vartxt = re.sub(r"\s+", " ", docline[1], flags=re.DOTALL).strip()
        docstrings[varname] = vartxt

    return docstrings


# noinspection PyShadowingBuiltins
def slice(input, output, depth):
    """
    Interpolate a ROMS dataset to a specific depth level

    More descriptive help
    :param input: Name of input file
    :param output: Name of output file
    :param depth: Depth levels, separated by comma
    """
    from .functions import open_roms, horz_slice
    import dask.diagnostics

    depths = [float(d) for d in depth.split(',')]

    with dask.diagnostics.ProgressBar():
        with open_roms(input) as dset_in:
            dset_out = horz_slice(dset_in, depths)
            dset_out.to_netcdf(output)


# noinspection PyShadowingBuiltins
def average(input, output, start, stop):
    """
    Interpolate a ROMS dataset to a specific depth level

    More descriptive help
    :param input: Name of input file
    :param output: Name of output file
    :param start: Start date
    :param stop: Stop date
    """
    from .functions import open_roms, average
    import dask.diagnostics
    import numpy as np

    start = np.datetime64(start)
    stop = np.datetime64(stop)

    with dask.diagnostics.ProgressBar():
        with open_roms(input) as dset_in:
            dset_out = average(dset_in, start, stop)
            dset_out.to_netcdf(output)
