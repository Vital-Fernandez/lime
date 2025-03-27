import logging
from pathlib import Path
from lime.io import load_cfg
from copy import deepcopy

_logger = logging.getLogger('LiMe')


def nested_dict(d, formatting_d):

    for key, value in d.items():
        if isinstance(value, dict):  # If the value is another dictionary, recurse
            nested_dict(value, formatting_d)
        else:
            d[key] = formatting_d.get(value, value)  # Otherwise, just print the value

    return d


def latex_science_float(f, dec=2):
    float_str = f'{f:.{dec}g}'
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str


def spectrum_figure_labels(units_wave, units_flux, norm_flux, plotting_library='matplotlib'):

    # Wavelength axis units
    x_label = units_wave.to_string('latex')
    x_label = f'Wavelength ({x_label})'

    # Flux axis units
    norm_flux = units_flux.scale if norm_flux is None else norm_flux
    norm_label = r'\right)$' if norm_flux == 1 else r' \,\cdot\,{}\right)$'.format(latex_science_float(1 / norm_flux))

    y_label = f"Flux {units_flux.to_string('latex')}"
    y_label = y_label.replace(r'$\mathrm{', r'$\left(')
    y_label = y_label.replace('}$', norm_label)

    if plotting_library == 'bokeh':
        x_label, y_label = x_label.replace('$', '$$'), y_label.replace('$', '$$')

    return x_label, y_label


class Themer:

    def __init__(self, conf, style=None):

        # Attributes
        self.conf = None        # All the formating data
        self.style = None       # Label of the active style
        self.active_conf = None   # Dictionary with the active figure configuration and library

        # LiMe plots personalization
        self.colors = None      # Features individual colors
        self.scale = None

        # LiMe plots personalization (library dependant)
        self.bokeh = None        # Features individual size
        self.plt = None          # Features individual size

        # Assign default
        self.conf = conf.copy()
        self.set_style(style)

        return


    @classmethod
    def from_toml(cls, fname, style=None):

        conf = load_cfg(fname, fit_cfg_suffix=None)

        return cls(conf, style)


    def fig_defaults(self, user_fig=None, fig_type=None, plot_lib='matplotlib'):

        # Get plot configuration
        if fig_type is None:
            fig_conf = self.active_conf[plot_lib]
        else:
            fig_conf = {** self.active_conf[plot_lib], **self.active_conf[f'{plot_lib}_{fig_type}']}

        # Get user configuration
        fig_conf = fig_conf if user_fig is None else {**fig_conf, **user_fig}

        return fig_conf


    def ax_defaults(self, user_ax, units_wave, units_flux, norm_flux, fig_type='default', plotting_library='matplotlib',
                    **kwargs):

        # Default wavelength and flux
        if fig_type == 'default':

            # Spectrum labels x-wavelegth, y-flux # TODO without units
            x_label, y_label = spectrum_figure_labels(units_wave, units_flux, norm_flux, plotting_library=plotting_library)
            ax_cfg = {'xlabel': x_label, 'ylabel': y_label}

            # Update with the user configuration
            ax_cfg = ax_cfg if user_ax is None else {**ax_cfg, **user_ax}

        # Spatial cubes
        elif fig_type == 'cube':

            ax_cfg = {} if user_ax is None else user_ax.copy()

            # Define the title
            if ax_cfg.get('title') is None:

                title = r'{} band'.format(kwargs['line_bg'].latex_label[0])

                line_fg = kwargs.get('line_fg')
                if line_fg is not None:
                    title = f'{title} with {line_fg.latex_label[0]} contours'

                if len(kwargs['masks_dict']) > 0:
                    title += f'\n and spatial masks at foreground'

                ax_cfg['title'] = title

            # Define x axis
            if ax_cfg.get('xlabel') is None:
                ax_cfg['xlabel'] = 'x' if kwargs['wcs'] is None else 'RA'

            # Define y axis
            if ax_cfg.get('ylabel') is None:
                ax_cfg['ylabel'] = 'y' if kwargs['wcs'] is None else 'DEC'

            # Update with the user configuration
            ax_cfg = ax_cfg if user_ax is None else {**ax_cfg, **user_ax}

        elif fig_type == 'velocity':

            x_label = 'Velocity (Km/s)'

            # Flux axis units
            norm_flux = units_flux.scale if norm_flux is None else norm_flux
            norm_label = r'\right)$' if norm_flux == 1 else r' \,\cdot\,{}\right)$'.format(latex_science_float(1/norm_flux))

            y_label = f"Flux {units_flux.to_string('latex')}"
            y_label = y_label.replace(r'$\mathrm{', r'$\left(')
            y_label = y_label.replace('}$', norm_label)

            ax_cfg = {'xlabel': x_label, 'ylabel': y_label}

            # Update with the user configuration
            ax_cfg = ax_cfg if user_ax is None else {**ax_cfg, **user_ax}

        # No labels
        else:
            ax_cfg = {}

            # Update with the user configuration
            ax_cfg = ax_cfg if user_ax is None else {**ax_cfg, **user_ax}


        return ax_cfg


    def set_style(self, style=None, scale=None, colors_conf=None, library=None):

        # Set the default style
        # self.style = ['default']
        #
        # # User requested style overwrite the default new style
        # if style is not None:
        #     self.style += [style] if isinstance(style, str) else style
        self.style = 'default' if style is None else style
        self.scale = ['default'] if style is None else [scale]

        # Set the library defaults
        self.active_conf = {'matplotlib': self.conf['matplotlib']['default'].copy(),
                            'bokeh': self.conf['bokeh']['default'].copy()}

        # Set the figure defaults
        for lib in ['matplotlib', 'bokeh']:
            if 'figure' in self.conf[lib]:
                for fig in self.conf[lib]['figure'].keys():
                    self.active_conf[f'{lib}_{fig}'] = self.conf[lib]['figure'][fig].copy()

        # Individual data features
        self.colors = self.conf['colors'][self.style].copy()

        # Figure colors for matplotlib
        for key, value in self.conf['matplotlib']['colors'].items():
            self.active_conf['matplotlib'][key] = self.colors.get(value, value)

        # Figure colors for bokeh
        colors_bokeh = nested_dict(deepcopy(self.conf['bokeh']['colors']), self.colors)
        self.active_conf['bokeh'].update(colors_bokeh)
                # for key, value in self.conf['bokeh']['colors'].items():
                #     self.active_conf['bokeh'][key] = self.colors.get(value, value)

        # Set the size
        if self.scale[0] in self.conf['matplotlib']['size']:
            self.plt = self.conf['matplotlib']['size'][self.scale[0]]

        if self.scale[0] in self.conf['bokeh']['size']:
            self.bokeh = self.conf['bokeh']['size'][self.scale[0]]


        return


# LiMe figure labels and color formatter
theme_file = Path(__file__).resolve().parent/'theme_lime.toml'
theme = Themer.from_toml(theme_file)