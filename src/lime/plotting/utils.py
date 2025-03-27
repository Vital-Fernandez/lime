import logging
from lime.transitions import Line
from matplotlib.pyplot import get_cmap
from matplotlib.colors import rgb2hex
from lime.io import _PARENT_BANDS

_logger = logging.getLogger('LiMe')


def parse_bands_arguments(label, log, ref_bands, norm_flux):

    line = None
    if label is None and (log.index.size > 0):
        label = log.index[-1]
        line = Line.from_log(label, log, norm_flux)

    # Line has been measured before
    elif label is not None and (log.index.size > 0):
        line = Line.from_log(label, log, norm_flux)

    # The user provided a reference band to check the region use it
    elif label is not None and ref_bands is not None:
        line = Line(label, ref_bands)

    elif label is not None and label in _PARENT_BANDS.index:
        line = Line(label, band=_PARENT_BANDS.loc[label, 'w1':'w6'].to_numpy())

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