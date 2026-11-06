import logging
_logger = logging.getLogger('LiMe')


from lime.transitions import Line
from lime.rsrc_manager import lineDB
from pathlib import Path

from inspect import signature, Signature, Parameter
from matplotlib.pyplot import get_cmap
from matplotlib.colors import rgb2hex
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from lime.tools import au



# Sentinel object for non input figures
_NO_FIG = object()


def latex_science_float(f, dec=2):
    float_str = f'{f:.{dec}g}'
    if "e" in float_str:
        base, exponent = float_str.split("e")
        exponent = int(exponent)
        if float(base) == 1:
            return rf"10^{{{exponent}}}"
        return rf"{base} \times 10^{{{exponent}}}"
    else:
        return float_str


def flux_label(units_flux, norm_flux):

    # Flux axis units
    if units_flux != au.dimensionless_unscaled:
        norm_flux = units_flux.scale if norm_flux is None else norm_flux

        # Unit without its scale
        unit_bare = au.CompositeUnit(1, units_flux.bases, units_flux.powers)
        unit_str = unit_bare.to_string('latex').replace(r'$\mathrm{', '').replace('}$', '')

        # Normalization first
        norm_str = '' if norm_flux == 1 else rf'{latex_science_float(norm_flux)}\,'

        y_label = rf'Flux $\left(\mathrm{{{norm_str}{unit_str}}}\right)$'
    else:
        y_label = 'Flux (scaled)'

    return y_label


def label_maker(plot_type, library='matplotlib', user_ax_cfg=None, **kwargs):

    # User input
    title = kwargs.get('title')

    match plot_type:

        # Wavelength vs flux
        case 'spectrum' | 'bands':
            x_label = f"Wavelength ({kwargs['units_wave'].to_string('latex')})"
            y_label = flux_label(kwargs['units_flux'], kwargs['norm_flux'])

        # Velocity vs flux
        case 'velocity':
            x_label = f"Velocity ({(au.km / au.s).to_string('latex')})"
            y_label = flux_label(kwargs['units_flux'], kwargs['norm_flux'])

        # Wavelength vs resolving power or pixel width
        case 'resolution':
            units_wave = kwargs['units_wave']
            x_label = f"Wavelength ({units_wave.to_string('latex')})"
            if kwargs.get('res_power_check'):
                y_label, default_title = r'Resolving power $R = \lambda/\Delta\lambda$', 'Spectral resolving power vs wavelength'
            else:
                y_label, default_title = f"Pixel width ({units_wave.to_string('latex')})", 'Pixel width vs wavelength'
            title = default_title if title is None else title

        case _:
            raise KeyError(f'Plot type is not recognized: {plot_type}')

    if library == 'matplotlib':
        default_labels = {'xlabel': x_label, 'ylabel': y_label, 'title': title}

        if user_ax_cfg is None:
            return default_labels
        else:
            return {**default_labels, **user_ax_cfg}

    if library == 'bokeh':
        x_label, y_label = x_label.replace('$', '$$'), y_label.replace('$', '$$')
        if user_ax_cfg is None:
            return x_label, y_label
        else:
            return user_ax_cfg.get('xlabel', x_label), user_ax_cfg.get('ylabel', y_label)

    return


def parse_bands_arguments(label, bands, log, norm_flux=None):

    line = None
    if label is None and (log.index.size > 0):
        label = log.index[-1]
        line = Line.from_transition(label, data_frame=log, norm_flux=norm_flux)

    # The user provided a reference band to check the region use it
    elif label is not None and bands is not None:
        line = Line.from_transition(label, data_frame=bands)

    # Line has been measured before
    elif label is not None and (log.index.size > 0):
        line = Line.from_transition(label, data_frame=log, norm_flux=norm_flux)

    elif label is not None and label in lineDB.frame.index:
        line = Line.from_transition(label)
        # line = Line(label, band=lineDB.frame.loc[label, 'w1':'w6'].to_numpy())

    else:
        _logger.warning(f'Line {label} has not been measured')

    return line


def color_selector(label, observations, idx_line, n_comps, scale_dict, colors_dict, library='matplotlib'):

    # Color and thickness
    if observations == 'no':

        # If only one component or combined
        if n_comps == 1:
            width_i, color = scale_dict['single_width'], colors_dict['profile']
            style = 'solid'

        # Component
        else:
            cmap = get_cmap(colors_dict['comps_map'])
            width_i, color = scale_dict['comp_width'], rgb2hex(cmap(idx_line/n_comps))
            style = 'dotted'

    # Case where the line has an error
    else:
        width_i, color = scale_dict['err_width'], 'red'
        style = 'solid'

    # Make dictionary with the params
    if library == 'matplotlib':
        line_format = dict(label=label, color=color, linestyle=style, linewidth=width_i)
    elif library == 'bokeh':
        label = '' if label is None else label
        line_format = dict(legend_label=label, line_color=color, line_dash=style, line_width=2)
    else:
        raise KeyError(f'Library {library} is not recognized, please use matplotlib or bokeh')

    return line_format


def window_maximize_positioning(maximize_check=False, center_check=False):

    if maximize_check:

        # Windows maximize
        mng = plt.get_current_fig_manager()

        try:
            mng.window.showMaximized()
        except:
            try:
                mng.resize(*mng.window.maxsize())
            except:
                _logger.debug(f'Unable to maximize the window')

    if center_check:

        try:
            mngr = plt.get_current_fig_manager()
            mngr.window.setGeometry(1100, 300, mngr.canvas.width(), mngr.canvas.height())
        except:
            _logger.debug(f'Unable to center plot window')

    return


class PlotAccessor:

    def __init__(self, obj):
        self._obj = obj
        self.fig = self.ax = None
        self._owns_fig = False

    @staticmethod
    def _bound(func, skip=1):

        """
        Decorator: the method takes the docstring and signature of `func`, without its first `skip` arguments
        (the ones the method supplies itself, e.g. the object or its frame).
        """

        def decorator(method):
            self_param = Parameter('self', Parameter.POSITIONAL_OR_KEYWORD)
            params = list(signature(func).parameters.values())[skip:]
            method.__signature__ = Signature([self_param, *params])
            method.__doc__ = func.__doc__
            return method

        return decorator


    def show(self, **kwargs):

        """
        Display the current Matplotlib figure.

        This is a convenience wrapper around :func:`matplotlib.pyplot.show`, used
        to display the most recently generated plot from the current plotting object
        (e.g., spectrum, bands, or velocity profile figures).

        Parameters
        ----------
        **kwargs
            Additional keyword arguments are passed directly to
            :func:`matplotlib.pyplot.show`. Common options include:

            - ``block`` : bool, optional
              If ``True`` (default), block execution until the plot window is closed.
              If ``False``, display the plot without blocking.

        Notes
        -----
        - This method should be called after generating a plot with other ``spec.plot`` functions if you want to render it interactively.
        - If you want to add your own artist to a LiMe figure you can set ``in_fig=None`` and then use ``spec.plot.show()``.

        Examples
        --------
        Generate a spectrum plot and plot it at a certain instance:

        >>> spec.plot.spectrum(in_fig=None)
        >>> spec.plot.show()

        Generate a spectrum plot, add some data and then open figure window:

        >>> spec.plot.spectrum(in_fig=None)
        >>> spec.plot.ax.scatter(w1, f1, label='Data point')
        >>> spec.plot.show()
        """

        plt.show(**kwargs)

        return


    def reset_figure(self):
        if self._owns_fig and isinstance(self.fig, Figure):
            plt.close(self.fig)

        self.fig = self.ax = None

        self._owns_fig = False

        return

    def save_close_fig_swicth(self, file_path=None, bbox_inches=None, fig_obj=None, maximise=False,
                              in_fig=_NO_FIG, in_ax=None):

        """
        Register who owns the figure, then show or save it accordingly.

        - ``in_ax`` given: the caller owns the axes and the figure, so nothing is shown, saved or closed.
        - ``in_fig`` given (even ``None``): the figure is kept open and not shown, so more artists can be added and
          ``plot.show()`` called later. If ``file_path`` is provided it is saved, but not closed.
        - Neither given: the figure is shown, or saved and closed if ``file_path`` is provided.

        ``self._owns_fig`` is True when the figure was created by the plotting function (``in_ax`` not given and
        ``in_fig`` None or not given), so the next ``reset_figure`` only closes figures created here.
        """

        # Check figure ownership first
        self._owns_fig = (in_ax is None) and ((in_fig is None) or (in_fig is _NO_FIG))

        # The caller owns the axes
        if in_ax is not None:
            return

        # The caller manages the figure display: only save it if requested
        external = in_fig is not _NO_FIG
        if external and file_path is None:
            return

        # Display
        if file_path is None:
            if bbox_inches is not None:
                fig_obj.tight_layout()
            window_maximize_positioning(maximise)
            plt.show()
            return

        # Save
        if not isinstance(file_path, (Path, str)):
            _logger.info(f'Output filepath is not recognized: {file_path} ({type(file_path)})')
            return

        fig_obj.savefig(file_path, bbox_inches=bbox_inches)
        if not external:  # only close the figures created here
            plt.close(fig_obj)

        return


    def save_fig(self, fname, bbox_inches='tight'):

        self.fig.savefig(fname, bbox_inches=bbox_inches)

        return