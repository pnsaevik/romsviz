from romsviz import scripts
from textwrap import dedent


class Test_get_argument_parser:
    def test_main_help_message(self):
        def funcname():
            """
            Sample function
            """

        parser = scripts.get_argument_parser([funcname])
        msg = parser.format_help()
        assert msg.strip() == dedent("""
            usage: romsviz [-h] [--version] [--verbose] subcmd ...

            Tools for ROMS plotting and data extraction

            optional arguments:
              -h, --help     show this help message and exit
              --version      show program's version number and exit
              --verbose, -v  increase verbosity

            subcommands:
              Run `romsviz (subcmd) --help` for more information about each subcommand

              subcmd
                funcname     Sample function
        """).strip()

    def test_subcommand_help_message(self):
        def myfunc(a, b):
            """
            Long description

            Over several lines
               with indentation

            :param a: First argument
            :param b: Second argument
            """
            return a + b

        parser = scripts.get_argument_parser([myfunc])
        msg = parser.romsviz_subparsers.choices['myfunc'].format_help()

        assert msg.strip() == dedent("""
            usage: romsviz myfunc [-h] a b

            Long description

            Over several lines
               with indentation

            positional arguments:
              a           First argument
              b           Second argument

            optional arguments:
              -h, --help  show this help message and exit
        """).strip()

    def test_returns_subcommand_arguments(self):
        def fn(a, b):
            return a + b

        parser = scripts.get_argument_parser([fn])
        args = parser.parse_args(['fn', '1', '2'])
        assert args.a == '1'
        assert args.b == '2'
        assert args.subcmd == 'fn'


class Test_first_line_of_docstring:
    def test_returns_only_first_line_of_docstring(self):
        def fn():
            """
            Sample function

            This is a longer description
            """

        txt = scripts.first_line_of_docstring(fn)
        assert txt == "Sample function"

    def test_returns_blank_text_if_no_docstring(self):
        txt = scripts.first_line_of_docstring(lambda: None)
        assert txt == ""


class Test_argument_docstring:
    def test_returns_blank_text_if_no_docstring(self):
        docstrings = scripts.argument_docstring(lambda a, b: None)
        assert docstrings['a'] == ""
        assert docstrings['b'] == ""

    def test_returns_argument_wise_docstrings(self):
        def fn(a, b, c):
            """
            Fine and nice function description
            :param a: This is argument a
            :param b: The parameter is b
            :param c: This is how you use c
            :return:
            """
            return a + b + c

        docstrings = scripts.argument_docstring(fn)
        assert docstrings['a'] == "This is argument a"
        assert docstrings['b'] == "The parameter is b"
        assert docstrings['c'] == "This is how you use c"
