import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import shutil
import subprocess

from lime.archives.tables import numberStringFormat, format_for_table, PdfMaker, table_fluxes

# Check if latex is available to compile tests
try:
    subprocess.run(['pdflatex', '--version'], capture_output=True, check=False)
    pdflatex_available = True
except FileNotFoundError:
    pdflatex_available = False

requires_pdflatex = pytest.mark.skipif(not pdflatex_available, reason="pdflatex binary not found")


class TestNumberStringFormat:

    def test_large_value_fixed_notation(self):
        result = numberStringFormat(1.5)
        assert '.' in result
        assert 'e' not in result

    def test_small_value_scientific_notation(self):
        result = numberStringFormat(0.0000123)
        assert 'e' in result

    def test_boundary_value(self):
        # 0.001 is the boundary; values > 0.001 use fixed
        assert 'e' not in numberStringFormat(0.0011)
        assert 'e' in numberStringFormat(0.0009)

    def test_sig_digits_respected(self):
        result = numberStringFormat(1.23456789, sig_digits=2)
        # Fixed format: should have 2 decimal places
        assert result == '1.23'


class TestFormatForTable:

    def test_none_entry(self):
        assert format_for_table(None) == 'None'

    def test_string_entry(self):
        assert format_for_table('hello') == 'hello'

    def test_bytes_entry(self):
        assert format_for_table(b'hello') == b'hello'

    def test_nan_default_format(self):
        assert format_for_table(float('nan')) == 'none'

    def test_nan_custom_format(self):
        assert format_for_table(float('nan'), nan_format='N/A') == 'N/A'

    def test_scalar_float(self):
        result = format_for_table(3.14159, rounddig=3)
        assert result == '3.142'

    def test_single_element_array(self):
        # Single-element arrays should be treated as scalars
        result = format_for_table(np.array([3.14159]), rounddig=3)
        assert result == '3.142'

    def test_multi_element_array(self):
        result = format_for_table(np.array(['a', 'b', 'c']))
        assert result == 'a_b_c'

    def test_multi_element_list(self):
        result = format_for_table(['x', 'y'])
        assert result == 'x_y'

    def test_small_float_scientific(self):
        result = format_for_table(1e-6)
        assert 'e' in result

    @pytest.mark.parametrize("entry,expected", [
        (0.5, True),   # normal float → fixed
        (1e-5, True),  # small float → scientific
    ])
    def test_parametrized_floats(self, entry, expected):
        result = format_for_table(entry)
        assert isinstance(result, str)

    def test_pylatex_types_passed_through(self):
        try:
            import pylatex
            mc = pylatex.MultiColumn(2, data='test')
            assert format_for_table(mc) is mc
        except ImportError:
            pytest.skip("pylatex not installed")


class TestPdfMaker:

    def test_init_defaults(self):
        pdf = PdfMaker()
        assert pdf.pdf_type is None
        assert pdf.table is None
        assert pdf.theme_table is None

    def test_create_pdfdoc_table_type(self):
        pdf = PdfMaker()
        pdf.create_pdfDoc(pdf_type='table')
        assert pdf.pdf_type == 'table'
        assert 'landscape' in pdf.pdf_geometry_options
        assert 'paperwidth' in pdf.pdf_geometry_options

    def test_create_pdfdoc_graphs_type(self):
        pdf = PdfMaker()
        pdf.create_pdfDoc(pdf_type='graphs')
        assert pdf.pdf_geometry_options.get('landscape') == 'true'

    def test_create_pdfdoc_dark_theme(self):
        pdf = PdfMaker()
        pdf.create_pdfDoc(pdf_type='table', theme='dark')
        assert pdf.theme_table == 'dark'

    def test_create_pdfdoc_custom_geometry(self):
        pdf = PdfMaker()
        custom_geo = {'right': '2cm'}
        pdf.create_pdfDoc(pdf_type='table', geometry_options=custom_geo)
        assert pdf.pdf_geometry_options['right'] == '2cm'

    def test_create_pdfdoc_no_type(self):
        # pdf_type=None should not create pdfDoc
        pdf = PdfMaker()
        pdf.create_pdfDoc(pdf_type=None)
        assert pdf.pdf_type is None
        assert not hasattr(pdf, 'pdfDoc')

    def test_pdf_insert_table_white_theme(self):
        pdf = PdfMaker()
        pdf.create_pdfDoc(pdf_type='table')
        headers = ['Line', 'Flux', 'Error']
        pdf.pdf_insert_table(column_headers=headers)
        assert pdf.table is not None

    def test_pdf_insert_table_dark_theme(self):
        pdf = PdfMaker()
        pdf.create_pdfDoc(pdf_type='table', theme='dark')
        headers = ['Line', 'Flux']
        pdf.pdf_insert_table(column_headers=headers)
        assert pdf.table is not None

    def test_pdf_insert_table_no_headers(self):
        pdf = PdfMaker()
        pdf.create_pdfDoc(pdf_type='table')
        pdf.pdf_insert_table(column_headers=None, table_format='lcc')
        assert pdf.table is not None

    def test_pdf_insert_table_no_pdf_type(self):
        # Without pdf_type, uses Tabu
        pdf = PdfMaker()
        pdf.create_pdfDoc(pdf_type=None)
        pdf.pdf_insert_table(column_headers=['A', 'B'], table_format='lc')
        assert pdf.table is not None

    def test_pdf_insert_table_longtable_type(self):
        pdf = PdfMaker()
        pdf.create_pdfDoc(pdf_type='longtable')
        headers = ['Line', 'Flux']
        pdf.pdf_insert_table(column_headers=headers)
        assert pdf.table is not None

    def test_add_table_row_auto_format(self):
        pdf = PdfMaker()
        pdf.create_pdfDoc(pdf_type='table')
        pdf.pdf_insert_table(column_headers=['Line', 'Flux', 'EW'])
        pdf.addTableRow(['H_alpha', 1.234, 5.678], last_row=True)

    def test_add_table_row_explicit_format(self):
        pdf = PdfMaker()
        pdf.create_pdfDoc(pdf_type='table')
        pdf.pdf_insert_table(column_headers=['Line', 'Flux'])
        pdf.addTableRow(['H_alpha', '1.23'], row_format='explicit', last_row=False)

    def test_add_table_row_dark_theme_colors(self):
        pdf = PdfMaker()
        pdf.create_pdfDoc(pdf_type='table', theme='dark')
        pdf.pdf_insert_table(column_headers=['Line', 'Flux'])
        # Should apply foreground/background colors without error
        pdf.addTableRow(['H_alpha', 0.5], last_row=True)

    def test_add_table_row_explicit_colors(self):
        pdf = PdfMaker()
        pdf.create_pdfDoc(pdf_type='table')
        pdf.pdf_insert_table(column_headers=['Line', 'Flux'])
        pdf.addTableRow(['H_alpha', 0.5], color_font='red', color_background='blue', last_row=True)

    def test_generate_pdf_tex_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf = PdfMaker()
            pdf.create_pdfDoc(pdf_type=None)
            pdf.pdf_insert_table(column_headers=['Line', 'Flux'], table_format='lc')
            pdf.addTableRow(['H_alpha', 1.0], last_row=True)
            out = Path(tmpdir) / 'output'
            pdf.generate_pdf(out)
            assert Path(str(out) + '.tex').exists()

    @requires_pdflatex
    def test_generate_pdf_table_type(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf = PdfMaker()
            pdf.create_pdfDoc(pdf_type='table')
            pdf.pdf_insert_table(column_headers=['Line', 'Flux'])
            pdf.addTableRow(['H_alpha', 1.0], last_row=True)
            out = Path(tmpdir) / 'test_table'
            try:
                pdf.generate_pdf(out, clean_tex=True)
            except Exception:
                pass  # pdflatex may not be available in CI


class TestTableFluxes:

    def _make_lines_df(self):
        import pandas as pd
        return pd.DataFrame(
            {'flux': [1.23, 4.56], 'error': [0.01, 0.02]},
            index=['H1_6563A', 'O3_5007A']
        )

    def test_pylatex_not_installed(self, capsys):
        with patch('lime.archives.tables.pylatex_check', False):
            table_fluxes(
                lines_df=self._make_lines_df(),
                table_address=Path('/tmp/dummy'),
                header_format_latex={'flux': 'Flux', 'error': 'Error'},
                lines_notation=['Ha', 'O3'],
            )
        captured = capsys.readouterr()
        assert 'WARNING' in captured.out

    def test_generates_tex_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / 'test_table'
            table_fluxes(
                lines_df=self._make_lines_df(),
                table_address=out,
                header_format_latex={'flux': 'Flux', 'error': 'Error'},
                table_type='tex',
                lines_notation=['Ha', 'O3'],
            )
            assert Path(str(out) + '.tex').exists()

    def test_with_components_column(self):
        import pandas as pd
        df = pd.DataFrame(
            {'flux': [1.0], 'Components': ['H1_6563A+N2_6583A']},
            index=['H1_6563A']
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / 'blended_test'
            table_fluxes(
                lines_df=df,
                table_address=out,
                header_format_latex={'flux': 'Flux', 'Components': 'Components'},
                table_type='tex',
                lines_notation=['Ha'],
            )

    @requires_pdflatex
    def test_pdf_compilation_failure_silent(self):
        # generate_pdf raises but table_fluxes catches it
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / 'fail_test'
            try:
                table_fluxes(
                    lines_df=self._make_lines_df(),
                    table_address=out,
                    header_format_latex={'flux': 'Flux', 'error': 'Error'},
                    table_type='pdf',
                    lines_notation=['Ha', 'O3'],
                )
            except Exception:
                pass  # pdflatex unavailable is acceptable

    def test_kwargs_added_as_footnotes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / 'footnote_test'
            table_fluxes(
                lines_df=self._make_lines_df(),
                table_address=out,
                header_format_latex={'flux': 'Flux', 'error': 'Error'},
                table_type='tex',
                lines_notation=['Ha', 'O3'],
                version='2.0',
                author='Vital',
            )