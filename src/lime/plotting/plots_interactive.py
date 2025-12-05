import logging
import numpy as np
import pandas as pd
from re import sub

from pathlib import Path
from matplotlib import pyplot as plt, rc_context
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.widgets import RadioButtons, SpanSelector, Slider
from matplotlib.ticker import NullLocator
from astropy.io import fits

from lime.io import load_frame, save_frame, LiMe_Error, check_file_dataframe
from lime.plotting.plots import Plotter, frame_mask_switch, save_close_fig_swicth, mplcursor_parser,\
                    determine_cube_images, load_spatial_mask, check_image_size, \
                    image_plot, spec_plot, spatial_mask_plot, _masks_plot, theme, line_band_plotter, spec_mask_plotter, \
                    line_band_scaler, spec_profile_plotter

from lime.tools import pd_get, unique_line_arr
from lime.transitions import label_decomposition, Line, check_continua_bands

_logger = logging.getLogger('LiMe')


def check_line_selection(spec, input_log, obj_bands, selected_by_default=True, **kwargs):

    # Use the reference bands (by default the lines database) to compute the lines on the object
    ref_params = {**kwargs, **{'automatic_grouping': False, 'fit_cfg':None,
                               'default_cfg_prefix': None, 'obj_cfg_prefix': None}}
    ref_bands = spec.retrieve.lines_frame(**ref_params)

    # Check if there is a physical bands file
    file_bands = check_file_dataframe(input_log, verbose=False)

    # Physical file has preference over ref bands and by default those are active
    if file_bands is not None:
        in_bands = file_bands
        default_status = 1

    # There is an input object bands and those are assumed as detected
    elif obj_bands is not None:
        in_bands = obj_bands
        default_status = 1

    # New bands are created and the user decides if detected or not
    else:
        in_bands = spec.retrieve.lines_frame(**kwargs)
        default_status = 1 if selected_by_default else 0

    # Extract the lines from each dataframe and only get the ones in common (without suffixes)
    in_lines = in_bands.index.to_numpy()
    ref_lines = ref_bands.index.to_numpy()

    in_core = np.array([sub(r'_(b|m)$', '', line) for line in in_lines])
    ref_core = np.array([sub(r'_(b|m)$', '', line) for line in ref_lines])

    # Give priority to those in the input log
    comb_lines, idx = np.unique(np.concatenate((in_core, ref_core)), return_index=True)
    idx_in = idx[idx < in_core.size]
    idx_ref = idx[idx >= in_core.size] - in_core.size

    # Create empty log
    labels_arr = np.concatenate((in_lines[idx_in], ref_lines[idx_ref]))
    log = pd.DataFrame(index=labels_arr, columns=in_bands.columns)

    # Fill the values
    ref_columns = np.intersect1d(log.columns, ref_bands.columns)
    log.loc[in_lines[idx_in], in_bands.columns] = in_bands.loc[in_lines[idx_in], in_bands.columns].to_numpy()
    log.loc[ref_lines[idx_ref], ref_columns] = ref_bands.loc[ref_lines[idx_ref], ref_columns].to_numpy()

    # Generate array with reference for reference
    active_lines = np.zeros(comb_lines.size).astype(int)
    active_lines[idx_in] = default_status

    # Sort to restore the order lost with unique
    if 'wavelength' in log.columns:
        sorted_indexes = log['wavelength'].values.argsort()
    else:
        wave_arr = label_decomposition(log.index.to_numpy(), params_list=['wavelength'], verbose=False)
        sorted_indexes = np.argsort(wave_arr[0])

    # Use the sorted index to reorder the DataFrame
    log = log.iloc[sorted_indexes]
    active_lines = active_lines[sorted_indexes].astype(bool)
    labels_arr = labels_arr[sorted_indexes]

    # Set NaN entries in dataframe as None
    if 'group_label' in log.columns:
        idcs_nan = log.group_label.isnull()
        log.loc[idcs_nan, 'group_label'] = 'none'

    return log, labels_arr,  active_lines


def load_redshift_table(file_address, column_name):

    file_address = Path(file_address)

    # Open the file
    if file_address.is_file():
        log = load_frame(file_address)
        if not (column_name in log.columns):
            _logger.info(f'No column "{column_name}" found in input dataframe, a new column will be added to the file')

    # Load the file
    else:
        if file_address.parent.is_dir():
            log = pd.DataFrame(columns=[column_name])
        else:
            raise LiMe_Error(f'The input log directory: {file_address.parent} \n does not exist')

    return log


def save_redshift_table(object, redshift, file_address):

    if redshift != 0:
        filePath = Path(file_address)

        if filePath.parent.is_dir():

            # Create a new dataframe and save it
            if not filePath.is_file():
                df = pd.DataFrame(data=redshift, index=[object], columns=['redshift'])

            # Replace or append to dataframe
            else:
                df = pd.read_csv(filePath, delim_whitespace=True, header=0, index_col=0)
                df.loc[object, 'redshift'] = redshift

            # Save back
            with open(filePath, 'wb') as output_file:
                string_DF = df.to_string()
                output_file.write(string_DF.encode('UTF-8'))

        else:
            _logger.warning(f'Output redshift table folder does not exist at {file_address}')

    return


def circle_band_label(current_label):
    match current_label[-2:]:
        case '_b':
            return f'{current_label[:-2]}_m'
        case '_m':
            return current_label[:-2]
        case _:
            return f'{current_label}_b'


def save_or_clear_log(log, log_address, active_lines, log_parameters='all'):

    if log_parameters is None:
        log_parameters = ['wavelength', 'wave_vac', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'latex_label',
                          'units_wave', 'particle', 'transition',  'rel_int']

    if np.sum(active_lines) == 0:
        if log_address.is_file():
            log_address.unlink()
    else:
        if log_address is not None:
            save_frame(log_address, log.loc[active_lines], parameters=log_parameters)
        else:
            _logger.warning(r"Not output redshift log provided, the selection won't be stored")

    return


class BandsInspection:

    def __init__(self):

        self.fig = None
        self.ax_list = None
        self.ax = None

        self.y_scale = None
        self.show_continua = None
        self.line_list = None
        self.active_lines = None
        self.fname = None

        self.line = None
        self.log = None
        self.mask = None

        self.wave_plot = None
        self.flux_plot = None
        self.z_corr = None
        self.idcs_mask = None

        self.color_bg = {True: theme.colors['line_selected'],
                         False: theme.colors['line_removed']}

        self.out_params = ["wavelength", "wave_vac", "w1", "w2", "w3", "w4", "w5", "w6",
                            "units_wave", "particle", "transition", "rel_int"]

        return

    def bands(self, fname, bands=None, default_status=True, show_continua=False, y_scale='auto', n_cols=6, n_rows=None,
              col_row_scale=(1, 0.5), fig_cfg=None, in_fig=None, maximize=False, **kwargs):

        """
        Launch an interactive line-bands editor and save selections to a lines frame file.

        This tool opens a  plot grid, one per spectral line, allowing you to
        **inspect** each region and **adjust the band central edges (w3–w4)** used for
        measurements. Edits are written to ``fname`` (a bands table on disk). If you
        run the editor again on the same file, **existing user selections are preserved**.

        A reference lines frame with the candidate lines can be provided via ``bands`` the default LiMe bands database is used.

        The function accepts the arguments from the :meth:`lime.Spectrum.retrieve.lines_frame` to adjust the default database
        values.

        If `show_continua=True`` the user can also adjust the continua bands region.

        **Left-click and drag** within a subplot to adjust the wavelength band limits interactively.

        A **middle-click** on a subplot to cycle through the line group types in the output bands file.
        The available suffixes are: blended (``_b``), merged (``_m``), and single (no suffix). The line label in the
        plot title updates accordingly.

        A **right-click** adds/remove a line from the selection (exluded lines have a red background)

        Parameters
        ----------
        fname : str or pathlib.Path
            Output file path where the edited bands table will be saved.
        bands : pandas.DataFrame, str, or pathlib.Path, optional
            Reference bands table (or path) providing initial band limits for each line.
            If ``None``, the default LiMe bands database is used. See
            :ref:`bands documentation <line-bands-doc>`.
        default_status : bool, optional
            Initial selection status to apply **only** when a line has no existing entry
            in ``fname`` (i.e., on first creation). Default is ``True``.
        show_continua : bool, optional
            If ``True``, draw the continuum side bands in each panel. Default ``False``.
        y_scale : {"auto", "linear", "log"}, optional
            Flux scale for all panels. ``"auto"`` chooses a sensible scale per panel;
            otherwise force Matplotlib’s ``"linear"`` or ``"log"``. Default ``"auto"``.
        n_cols : int, optional
            Number of columns in the grid. Default ``6``.
        n_rows : int, optional
            Number of rows in the grid. If ``None``, it is inferred from the number of lines.
        col_row_scale : tuple of (float, float), optional
            Multiplicative factors for panel width and height (rough layout scaling).
            Default ``(1, 0.5)``.
        fig_cfg : dict, optional
            Matplotlib figure configuration (e.g., size, DPI). See `matplotlib.RcParams
            <https://matplotlib.org/stable/api/matplotlib_configuration_api.html#matplotlib.RcParams>`_.
        in_fig : matplotlib.figure.Figure, optional
            Existing figure to plot into. If ``None``, a new figure is created.
        maximize : bool, optional
            If ``True``, maximize the window after rendering. Default ``False``.
        **kwargs
            Additional keyword arguments passed to :meth:`lime.Spectrum.retrieve.lines_frame`

        Notes
        -----
        - Band edits (w1–w6) are written to ``fname`` immediately when saved/closed.
        - If no changes are performed the output ``fname`` will not be created.
        - Existing entries in ``fname`` are **not overwritten**; new lines inherit
          their initial state from ``default_status``.
        - **Left-click and drag** within a subplot to adjust the wavelength band limits.
        - **Right-click** on a line subplot to include or exclude the line from the output bands file.
            Excluded lines are displayed with a red background.
        - **Middle-click** on a subplot to cycle through the line group types in the output bands file.
            The available suffixes are: blended (``_b``), merged (``_m``), and single (no suffix).
            The line label in the plot title updates accordingly.

        Examples
        --------
        Start from the default database and save to a new file:

        >>> spec.check.bands("my_bands.xlsx")

        Initialize from an existing bands table and show the continua bands for the selection:

        >>> spec.check.bands("session_bands.xlsx", bands="ref_bands.xlsx", show_bands=True)

        Adjusting the initial ``bands`` generation with the arguments from the  with the
        :meth:`lime.Spectrum.retrieve.lines_frame` function

        >>> gp_spec.check.bands(lineBandsFile, band_vsigma=100, n_sigma=4, instrumental_correction=True,
        >>>                     map_band_vsigma={'H1_4861A': 200, 'H1_6563A': 200, 'O3_4959A': 250, 'O3_5007A': 250},
        >>>                     fit_cfg=obs_cfg, ref_bands=osiris_gp_df_path, show_bands=True, maximize=True)

        """

        # Declare the function attributes
        self.y_scale = y_scale
        self.show_continua = show_continua

        # Check the address of the output line frame
        if isinstance(fname, (str, Path)):
            self.fname = Path(fname)
            if not self.fname.parent.is_dir():
                raise LiMe_Error(f'Input bands file directory does not exist ({self.fname.parent.as_posix()})')

        # Establish the list of line and their status
        self.log, self.line_list, self.active_lines = check_line_selection(self._spec, self.fname, bands,
                                                                           default_status, **kwargs)

        # Store spectrum
        self.wave_plot, self.flux_plot, _, self.z_corr, self.idcs_mask = frame_mask_switch(self._spec, True)

        # Check there are lines in selection
        n_lines = self.log.index.size
        if n_lines > 0:

            # Compute the rows and columns number
            if n_lines > n_cols:
                n_rows = n_rows or int(np.ceil(n_lines / n_cols))
            else:
                n_cols, n_rows = n_lines, 1
            n_grid = n_cols * n_rows

            # User configuration overwrites default configuration
            size_conf = {'figure.figsize': (n_cols * col_row_scale[0], n_rows * col_row_scale[1])}
            size_conf = size_conf if fig_cfg is None else {**size_conf, **fig_cfg}
            plt_cfg = theme.fig_defaults(size_conf, fig_type='grid')

            # Launch the interative figure
            with rc_context(plt_cfg):

                # Figure structure
                self.fig = plt.figure() if in_fig is None else in_fig
                # grid_spec = self.fig.add_gridspec(2, 1, height_ratios=[1, 0.1])
                grid_spec = self.fig.add_gridspec(1, 1)
                gs_lines = grid_spec[0].subgridspec(n_rows, n_cols, hspace=0.5)
                self.ax_list = gs_lines.subplots().flatten() if n_lines > 1 else [gs_lines.subplots()]

                # Fill the plot axes
                span_selector_dict = {}
                for i in range(n_grid):
                    if i < n_lines:
                        self.line = self.line_list[i]
                        self.plot_line_BI(self.ax_list[i], self.line)
                        span_selector_dict[f'spanner_{i}'] = SpanSelector(self.ax_list[i], self.on_select_BI, button=1,
                                                                          direction='horizontal', useblit=True,
                                                                          props=dict(alpha=0.5, facecolor='tab:blue'))
                    else:
                        # Clear not filled axes
                        self.fig.delaxes(self.ax_list[i])

                # Connecting the figure to the interactive widgets
                self.fig.canvas.mpl_connect('button_press_event', self.on_click_BI)
                self.fig.canvas.mpl_connect('axes_enter_event', self.on_enter_axes_BI)

                # Show the image
                save_close_fig_swicth(None, True, self.fig, maximise=maximize,
                                      plot_check=True if in_fig is None else False)

        else:
            _logger.warning(f'No lines found in the lines mask for the object wavelentgh range')

        return

    def plot_line_BI(self, ax, line, scale_dict=theme.plt):

        # Establish the limits for the line spectrum plot
        mask = self.log.loc[line, 'w1':'w6'].to_numpy() * self.z_corr
        idcs_band = np.searchsorted(self.wave_plot, mask)

        # Review the bands edges
        idcs_band = check_continua_bands(idcs_band, self.wave_plot, reset_w2_w5=True)

        # Just the center region is adjusted
        if self.show_continua:
            idxL = idcs_band[2] - 10 if idcs_band[2] - 10 > 0 else 0
            idxH = idcs_band[3] + 10 if idcs_band[3] + 10 < self.wave_plot.size - 1 else self.wave_plot.size - 1

        # Center + continua
        else:
            idxL = idcs_band[0] - 5 if idcs_band[0] - 5 > 0 else 0
            idxH = idcs_band[5] + 5 if idcs_band[5] + 5 < self.wave_plot.size - 1 else  self.wave_plot.size - 1

        # Plot the spectrum
        ax.step(self.wave_plot[idxL:idxH]/self.z_corr, self.flux_plot[idxL:idxH]*self.z_corr, where='mid',
                color=theme.colors['fg'], linewidth=scale_dict['spectrum_width'])

        # Continuum bands
        line_band_plotter(ax, self.wave_plot, self.flux_plot, self.z_corr, idcs_band, line, theme.colors,
                          show_adjacent=self.show_continua)

        # Plot the masked pixels
        spec_mask_plotter(ax, self.idcs_mask[idxL:idxH], self.wave_plot[idxL:idxH], self.flux_plot[idxL:idxH],
                          self.z_corr, self.log, line, theme.colors)

        # Plot line location
        wave_line = pd_get(self.log, line, 'wavelength')
        if wave_line is not None:
            ax.axvline(wave_line, linestyle='--', color='grey', linewidth=0.5)

        # Background for selective line for selected lines
        if self.active_lines[self.line_list == line][0]:
            ax.set_facecolor(theme.colors['line_selected'])
        else:
            ax.set_facecolor(theme.colors['line_removed'])

        # Scale the y axis
        line_band_scaler(ax, self.flux_plot[idxL:idxH] * self.z_corr, 'auto')

        # Formatting the figure
        ax.set_title(line, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.yaxis.set_minor_locator(NullLocator())
        ax.get_xlim() # TODO without this one there is no plot

        return

    def on_select_BI(self, w_low, w_high):

        # Check we are not just clicking on the plot
        if w_low != w_high:

            # Just the central bands
            if self.show_continua is False:
                self.log.at[self.line, 'w3'] = w_low
                self.log.at[self.line, 'w4'] = w_high

                # Move the other bands to avoid issues
                idx_low, idx_high = np.searchsorted(self.wave_plot, (w_low, w_high))
                self.log.at[self.line, 'w1'] = self.wave_plot[np.max((0, idx_low - 5))]
                self.log.at[self.line, 'w2'] = self.wave_plot[np.max((0, idx_low - 1))]
                self.log.at[self.line, 'w5'] = self.wave_plot[np.min((idx_high + 1, self.wave_plot.size -1))]
                self.log.at[self.line, 'w6'] = self.wave_plot[np.min((idx_high + 5, self.wave_plot.size -1))]

            # Central and adjacent bands
            else:

                # Correcting line band
                if w_low > self.log.at[self.line, 'w2'] and w_high <self.log.at[self.line, 'w5']:
                    self.log.at[self.line, 'w3'] = w_low
                    self.log.at[self.line, 'w4'] = w_high

                # Correcting blue band
                elif w_low < self.log.at[self.line, 'w3'] and w_high < self.log.at[self.line, 'w3']:
                    self.log.at[self.line, 'w1'] = w_low
                    self.log.at[self.line, 'w2'] = w_high

                # Correcting Red
                elif w_low > self.log.at[self.line, 'w4'] and w_high >  self.log.at[self.line, 'w4']:
                    self.log.at[self.line, 'w5'] = w_low
                    self.log.at[self.line, 'w6'] = w_high

                # Removing line
                elif w_low < self.log.at[self.line, 'w1'] and w_high > self.log.at[self.line, 'w6']:
                    print(f'\n-- The line {self.line} mask has been removed')

                # Weird case
                else:
                    _logger.info(f'Unsuccessful line selection: {self.line}: w_low: {w_low}, w_high: {w_high}')

            # Save the log to the file
            save_or_clear_log(self.log, self.fname, self.active_lines, self.out_params)

            # Redraw the line measurement
            self.ax.clear()
            self.plot_line_BI(self.ax, self.line)
            self.fig.canvas.draw()

        return

    def on_enter_axes_BI(self, event):

        # Assign current line and axis
        self.ax = event.inaxes
        title = self.ax.get_title()
        if title != '':
            self.line = title

    def on_click_BI(self, event):

        if event.button in (2, 3):

            # Update the line label
            if event.button == 2:
                idx = self.line_list == self.line
                new_name = circle_band_label(self.line)
                self.log.rename(index={self.line: new_name}, inplace=True)
                self.line = new_name
                self.line_list[idx] = new_name

                # Remove group label and latex label since it cannot be restored
                for entry in ['group_label', 'latex_label']:
                    if entry in self.log.columns:
                        self.log.loc[self.line, entry] = 'none'

            # Update the line active status
            if event.button == 3:
                idx = self.line == self.line_list
                self.active_lines[idx] = np.invert(self.active_lines[idx])

            # Save the log to the file
            save_or_clear_log(self.log, self.fname, self.active_lines, self.out_params)

            # Plot the line selection with the new Background
            self.ax.clear()
            self.plot_line_BI(self.ax, self.line)
            self.fig.canvas.draw()

        return


class RedshiftInspection:

    def __init__(self):

        # Plot Attributes
        self._fig = None
        self._ax = None
        self._AXES_CONF = None
        self._spec_label = None
        self._legend_handle = None

        # Input data
        self._obj_idcs = None
        self._column_log = None
        self._log_address = None
        self._output_idcs = None
        self._latex_array = None
        self._waves_array = None

        # User pointing
        self._lineSelection = None
        self._user_point = None
        self._none_value = None
        self._unknown_value = None
        self._sample_object = None

    def redshift(self, obj_idcs, reference_lines, output_file_log=None, output_idcs=None, redshift_column='redshift',
                 initial_z=None, none_value=np.nan, unknown_value=0.0,  legend_handle='levels', maximize=False, title=None,
                 output_address=None, n_pixels=10, fig_cfg={}, ax_cfg={}, in_fig=None, **kwargs):


        # Check if input tuple
        if isinstance(obj_idcs, tuple):
            obj_idcs = pd.MultiIndex.from_tuples([obj_idcs], names=self._sample.index.names)

        # Assign the attributes
        self._obj_idcs = obj_idcs if isinstance(obj_idcs, pd.MultiIndex) else self._sample.loc[obj_idcs].index
        self._column_log = redshift_column
        self._none_value = none_value
        self._unknown_value = unknown_value
        self._spec_label = "" if title is None else title
        self._legend_handle = legend_handle
        self._user_point = None

        # Parameters for the load function
        self._load_params = {**self._sample.load_params, **kwargs}
        self._load_params['redshift'] = 0

        # Output Log params
        self._log_address = output_file_log

        # Only save new redshift in input idx if None provided
        if output_idcs is None:
            self._output_idcs = self._obj_idcs
        else:
            self._output_idcs = output_idcs if isinstance(output_idcs, pd.MultiIndex) else self._sample.loc[output_idcs].index

        # Check the redshift column exists
        if self._column_log not in self._sample.frame.columns:
            raise LiMe_Error(f'Redshift column "{redshift_column}" does not exist in the current sample log.')

        # Use provided redshift value
        if initial_z is not None:
            redshift_pred = initial_z
        else:
            redshift_pred = self._sample.loc[self._obj_idcs, self._column_log].to_numpy()
            redshift_pred = None if np.all(pd.isnull(redshift_pred)) else np.nanmean(redshift_pred)

        # Create initial entry
        self._compute_redshift(redshift_output=redshift_pred)

        # Get the lines transitions and latex labels
        reference_bands_df = check_file_dataframe(reference_lines)
        if reference_bands_df is None:
            raise LiMe_Error(f'Reference line log could not be read ({reference_lines})')
        else:
            if isinstance(reference_bands_df, pd.DataFrame):
                reference_lines = reference_bands_df.index.to_numpy()

        # Sort by wavelength
        _waves_array, _latex_array = label_decomposition(reference_lines, params_list=('wavelength', 'latex_label'))
        idcs_sorted = np.argsort(_waves_array)
        self._waves_array, self._latex_array = _waves_array[idcs_sorted], _latex_array[idcs_sorted]

        # Set the plot format where the user's overwrites the default
        size_conf = {'figure.figsize': (10, 6), 'axes.labelsize': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10}
        size_conf = size_conf if fig_cfg is None else {**size_conf, **fig_cfg}

        PLT_CONF = theme.fig_defaults(size_conf)
        self._AXES_CONF = theme.ax_defaults(ax_cfg, self._sample, fig_type=None)

        # Create and fill the figure
        with rc_context(PLT_CONF):

            # Generate the figure object and figures
            self._fig = plt.figure() if in_fig is None else in_fig
            gs = GridSpec(nrows=1, ncols=2, figure=self._fig, width_ratios=[2, 0.5], height_ratios=[1])
            self._ax = self._fig.add_subplot(gs[0])
            self._ax.set(**self._AXES_CONF)

            # Create the RadioButtons widget for the lines
            buttoms_ax = self._fig.add_subplot(gs[1])
            labels_buttons = [r'$None$'] + list(self._latex_array) + [r'$Unknown$']
            radio_props = {'s': [10] * len(labels_buttons)}
            label_props = {'fontsize': [6] * len(labels_buttons)}
            radio = RadioButtons(buttoms_ax, labels_buttons, radio_props=radio_props, label_props=label_props)

            # Plot the spectrum
            self._launch_plots_ZI()

            # Connect the widgets
            radio.on_clicked(self._button_ZI)
            self._fig.canvas.mpl_connect('button_press_event', self._on_click_ZI)

            # Plot on screen unless an output address is provided
            # save_close_fig_swicth(output_address, 'tight', self._fig, maximise=maximize)
            save_close_fig_swicth(None, None, self._fig, maximise=maximize,
                                  plot_check=True if in_fig is None else False)


        return

    def _launch_plots_ZI(self):

        # Get redshift from log
        redshift_pred = self._sample.loc[self._obj_idcs, self._column_log].to_numpy()
        redshift_pred = None if np.all(pd.isnull(redshift_pred)) else np.nanmean(redshift_pred)

        # Store the figure limits
        xlim, ylim = self._ax.get_xlim(), self._ax.get_ylim()

        # Redraw the figure
        self._ax.clear()
        self._plot_spectrum_ZI(self._ax)
        self._plot_line_labels_ZI(self._ax, self._user_point, redshift_pred)
        self._ax.legend(loc=4)

        title = f'{self._spec_label} z calculation'
        if redshift_pred not in [None, self._none_value, self._unknown_value]:
            title += f', redshift = {redshift_pred:0.3f}'
        self._ax.set_title(title)

        # Reset axis format
        if (xlim[0] != 0) and (xlim[0] != 1): # First time
            self._ax.set_xlim(xlim)
            self._ax.set_ylim(ylim)
        self._ax.set(**self._AXES_CONF)
        self._fig.canvas.draw()

        return

    def _plot_spectrum_ZI(self, ax):

        # Loop through the objects
        for i, obj_idx in enumerate(self._obj_idcs):

            # Load the spectrum with a zero redshift
            spec = self._sample.load_function(self._sample.frame, obj_idx, self._sample.file_address,
                                              instrument=self._sample.instrument, **self._load_params)

            # Plot on the observed frame with reshift = 0
            wave_plot, flux_plot, err_plot, z_corr, idcs_mask = frame_mask_switch(spec, True)

            # Plot the spectrum
            ax.step(wave_plot/z_corr, flux_plot*z_corr, label=self._label_generator(obj_idx), where='mid',
                    linewidth=theme.plt['spectrum_width'])

            # Plot the masked pixels
            _masks_plot(ax, None, wave_plot, flux_plot, z_corr, spec.frame, idcs_mask, color_dict=theme.colors)

        return

    def _plot_line_labels_ZI(self, ax, click_coord, redshift_pred):

        if (redshift_pred != 0) and (not pd.isnull(redshift_pred)):
            wave_min, wave_max = None, None
            for obj_idx in self._obj_idcs:

                # Load the spectrum
                spec = self._sample.load_function(self._sample.frame, obj_idx, self._sample.file_address,
                                                  instrument=self._sample.instrument, **self._load_params)

                wavelength = spec.wave.data
                wavelength = wavelength[~np.isnan(wavelength)]

                if wave_min is None:
                    wave_min = wavelength[0]
                else:
                    wave_min = wavelength[0] if wavelength[0] < wave_min else wave_min

                if wave_max is None:
                    wave_max = wavelength[-1]
                else:
                    wave_max = wavelength[-1] if wavelength[-1] > wave_max else wave_max

            # Check the lines which fit in the plot region
            idcs_in_range = np.logical_and(self._waves_array * (1 + redshift_pred) >= wave_min,
                                           self._waves_array * (1 + redshift_pred) <= wave_max)

            # Plot lines in region
            linesRange = self._waves_array[idcs_in_range]
            latexRange = self._latex_array[idcs_in_range]
            for i, lineWave in enumerate(linesRange):
                if latexRange[i] == self._lineSelection:
                    color_line = 'tab:red'
                else:
                    color_line = theme.colors['fg']

                ax.axvline(x=lineWave * (1 + redshift_pred), color=color_line, linestyle='--', linewidth=0.5)

                ax.annotate(latexRange[i], xy=(lineWave * (1 + redshift_pred), 0.85),
                            horizontalalignment="center",
                            rotation=90,
                            backgroundcolor='w',
                            size=6,
                            xycoords='data', xytext=(lineWave * (1 + redshift_pred), 0.85), textcoords=("data",
                                                                                                        "axes fraction"),
                            bbox=dict(
                                facecolor=theme.colors['bg'],  # Background color
                                edgecolor='none',  # Border color
                            ))

        return

    def _compute_redshift(self, redshift_output=None):

        # Routine not to overwrite first measurement
        if redshift_output is None:

            # First time case: Both input but be provided
            if self._lineSelection is not None:

                # Default case nothing is selected:
                if self._lineSelection == r'$None$':
                    _redshift_pred = self._none_value

                elif self._lineSelection == r'$Unknown$':
                    _redshift_pred = self._unknown_value

                # Wavelength selected
                else:
                    if self._user_point is not None:
                        idx_line = self._latex_array == self._lineSelection
                        ref_wave = self._waves_array[idx_line][0]
                        _redshift_pred = self._user_point[0] / ref_wave - 1

                    # Special cases None == NaN, Unknown == 0
                    else:
                        _redshift_pred = self._none_value
            else:
                _redshift_pred = self._none_value
        else:
            _redshift_pred = redshift_output

        # Store the new redshift

        self._sample.loc[self._output_idcs, self._column_log] = _redshift_pred

        # Save to file if provided
        if self._log_address is not None:
            save_frame(self._log_address, self._sample.frame)
        return

    def _button_ZI(self, line_selection):

        # Button selection
        self._lineSelection = line_selection

        # Compute the redshift
        self._compute_redshift()

        # Replot the figure
        self._launch_plots_ZI()

        return

    def _on_click_ZI(self, event, tolerance=3):

        if event.button == 2:

            self._user_point = (event.xdata, 0.5)

            # Compute the redshift
            self._compute_redshift()

            # Replot the figure
            self._launch_plots_ZI()

        return

    def _label_generator(self, idx_sample):

        if self._legend_handle == 'levels':

            spec_label =", ".join(map(str, idx_sample))

        else:

            if self._legend_handle in self._sample.index.names:
                idx_item = list(self._sample.index.names).index(self._legend_handle)
                spec_label = idx_sample[idx_item]

            elif self._legend_handle in self._sample.frame.columns:
                spec_label = self._sample.frame.loc[idx_sample, self._legend_handle]

            else:
                raise LiMe_Error(f'The input handle "{self._legend_handle}" is not found on the sample log columns')


        return spec_label


class CubeInspection:

    def __init__(self):

        # Data attributes
        self.grid_mesh = None
        self.bg_image = None
        self.fg_image = None
        self.fg_levels = None
        self.hdul_linelog = None
        self.ext_log = None
        self.spaxel_button = None
        self.add_remove_button = None
        self.spec = None

        # Mask correction attributes
        self.mask_file = None
        self.mask_ext = None
        self.masks_dict = {}
        self.mask_color = None
        self.mask_array = None

        # Plot attributes
        self.in_ax = None
        self.axes_conf = {}
        self.axlim_dict = {}
        self.color_norm = None
        self.mask_color_i = None
        self.key_coords = None
        self.marker = None
        self.rest_frame = None
        self.log_scale = None
        self.restore_zoom = False
        self.maintain_y_zoom = False

        return

    def cube(self, line_bg, bands=None, line_fg=None, min_pctl_bg=60, cont_pctls_fg=(90, 95, 99), bg_cmap='gray',
             fg_cmap='viridis', bg_norm=None, fg_norm=None, masks_file=None, masks_cmap='viridis_r', masks_alpha=0.2,
             rest_frame=False, log_scale=False, fig_cfg=None, ax_cfg_image=None, ax_cfg_spec=None, in_fig=None,
             fname=None, ext_frame_suffix='_LINELOG', maintain_y_zoom=True, wcs=None, spaxel_selection_button=1,
             add_remove_button=3, maximize=False):

        """
        Open an interactive cube viewer: image map (left) + spaxel spectrum (right).

        The left panel shows a band-summed flux map for ``line_bg`` (bands from ``bands`` or
        the default database). Optionally overlay **foreground contours** from ``line_fg``.
        Clicking on a spaxel updates the right panel with that spaxel’s spectrum. If
        a ``lines_file`` FITS log is provided, fitted profiles for the selected spaxel
        are also displayed.

        If a mask file is supplied (``masks_file``), a radio selector appears to toggle
        which binary mask is active/visible. You can add/remove the currently selected
        spaxel to/from the active mask using a configurable mouse button.

        Parameters
        ----------
        line_bg : str
            Line label for the background image (band-summed flux). See :ref:`bands documentation <line-bands-doc>`.
        bands : pandas.DataFrame, str, or pathlib.Path, optional
            Bands table (or path). If ``None``, the default LiMe bands database is used.
        line_fg : str, optional
            Line label for foreground **contours**. If provided, contours are computed
            from that line’s band-summed flux using ``cont_pctls_fg``.
        min_pctl_bg : float, optional
            Minimum percentile of the background band-summed flux used to set the lower
            display limit when ``bg_norm`` is not supplied. Default is ``60``.
        cont_pctls_fg : tuple of float, optional
            Sorted percentiles for foreground contours. Default is ``(90, 95, 99)``.
        bg_cmap : str, optional
            Colormap name for the background image. Default is ``"gray"``.
        fg_cmap : str, optional
            Colormap name for the foreground contours. Default is ``"viridis"``.
        bg_norm : matplotlib.colors.Normalize, optional
            Normalization for the background image. If ``None``, a symmetric-log style
            normalization is used (sensible defaults for wide dynamic ranges).
        fg_norm : matplotlib.colors.Normalize, optional
            Normalization for the foreground contours. If ``None``, a logarithmic
            normalization is used.
        masks_file : str or pathlib.Path, optional
            Path to a FITS file with binary spatial masks to overlay and edit.
        masks_cmap : str, optional
            Colormap used to render masks. Default is ``"viridis_r"``.
        masks_alpha : float, optional
            Alpha transparency for mask overlays (0–1). Default is ``0.2``.
        rest_frame : bool, optional
            If ``True``, show the spectrum panel in rest-frame wavelengths. Default ``False``.
        log_scale : bool, optional
            If ``True``, plot the spectrum panel with a logarithmic flux scale. Default ``False``.
        in_fig : matplotlib.figure.Figure, optional
            Existing figure to plot into. If ``None``, a new figure is created.
        fig_cfg : dict, optional
            Matplotlib figure configuration (e.g., size, DPI). See `matplotlib.RcParams
            <https://matplotlib.org/stable/api/matplotlib_configuration_api.html#matplotlib.RcParams>`_.
        ax_cfg_image : dict, optional
            Axes label/title overrides for the **image** panel; keys may include
            ``"xlabel"``, ``"ylabel"``, and ``"title"``.
        ax_cfg_spec : dict, optional
            Axes label/title overrides for the **spectrum** panel; keys may include
            ``"xlabel"``, ``"ylabel"``, and ``"title"``.
        fname : str or pathlib.Path, optional
            Path to a FITS lines-log file. If provided, fitted profiles for the selected
            spaxel are displayed when available. Each spaxel page must be named
            ``"{j}-{i}{ext_frame_suffix}"`` (e.g., ``"25-30_LINELOG"``).
        ext_frame_suffix : str, optional
            Suffix used to match spaxel pages inside ``lines_file``. Default ``"_LINELOG"``.
        maintain_y_zoom : bool, optional
            If ``True`` (default), preserve the current y-axis zoom on the spectrum panel
            when selecting different spaxels; if ``False``, autoscale on each selection.
        wcs : astropy.wcs.WCS, optional
            WCS to use for the image panel. If ``None``, the cube’s WCS is used when available.
            See `Astropy WCS <https://docs.astropy.org/en/stable/wcs/index.html>`_.
        spaxel_selection_button : int, optional
            Mouse button used to **select/preview** a spaxel on the image panel.
            Default is ``1`` (LEFT). Matplotlib button mapping: LEFT=1, MIDDLE=2, RIGHT=3,
            BACK=8, FORWARD=9.
        add_remove_button : int, optional
            Mouse button used to **add/remove** the selected spaxel to/from the active mask.
            Default is ``3`` (RIGHT). Matplotlib button mapping: LEFT=1, MIDDLE=2, RIGHT=3,
            BACK=8, FORWARD=9.
        maximize : bool, optional
            If ``True``, maximize the window after rendering. Default ``False``.

        Notes
        -----
        - **Interactivity:**
          - Click the image panel with ``spaxel_selection_button`` to update the spectrum panel.
          - Click the image panel with ``add_remove_button`` to toggle the selected spaxel in the active mask (when ``masks_file`` is provided).
          - A round button selection changes the active mask.
        - **Fitted profiles:** If :mod:`mplcursors` is installed and a fitted profile is
          displayed, left-click shows fit parameters and right-click removes the tooltip.
        - **Dynamic range tip:** With large flux dynamic ranges, percentile-based contour
          levels may appear non-linear in visual spacing.
        - **WCS handling:** If a valid WCS is provided (or available from the cube),
          the image panel uses WCS projection and labeled axes.

        Examples
        --------
        Basic interactive viewer with contours:

        >>> cube.plot.cube("O3_5007A", line_fg="H1_6563A")

        Use a masks file and RIGHT-click to add/remove spaxels from the active mask:

        >>> cube.plot.cube("H1_6563A", masks_file="halpha_masks.fits", add_remove_button=3)

        Show fitted profiles from a measurements log:

        >>> cube.plot.cube("O3_5007A", fname="linelog.fits", ext_frame_suffix="_LINELOG")
        """

        self.ext_log = ext_frame_suffix
        self.mask_file = masks_file
        self.spaxel_button = spaxel_selection_button
        self.add_remove_button = add_remove_button
        self.maintain_y_zoom = maintain_y_zoom
        self.bg_color, self.fg_color = bg_cmap, fg_cmap
        self.mask_color, self.mask_alpha = masks_cmap, masks_alpha
        self.rest_frame, self.log_scale = rest_frame, log_scale

        # Prepare the background image data
        line_bg, self.bg_image, self.bg_levels, self.bg_scale = determine_cube_images(self._cube, line_bg, bands,
                                                                                      min_pctl_bg, bg_norm,
                                                                                      contours_check=False)

        # Prepare the foreground image data
        line_fg, self.fg_image, self.fg_levels, self.fg_scale = determine_cube_images(self._cube, line_fg, bands,
                                                                                      cont_pctls_fg, fg_norm,
                                                                                      contours_check=True)

        # Mesh for the contours
        self.fg_mesh = None if line_fg is None else np.meshgrid(np.arange(0, self.fg_image.shape[1]),
                                                                np.arange(0, self.fg_image.shape[0]))

        # Load the masks
        self.masks_dict = load_spatial_mask(self.mask_file)
        self.mask_ext = list(self.masks_dict.keys())[0]  if len(self.masks_dict) > 0 else self.ext_log

        # Check that the images have the same size
        check_image_size(self.bg_image, self.fg_image, self.masks_dict)

        # Use the input wcs or use the parent one
        wcs = self._cube.wcs if wcs is None else wcs
        slices = None if wcs is None else ('x', 'y', 1) if wcs.naxis == 3 else ('x', 'y')

        # Use central voxel as initial coordinate
        self.key_coords = int(self._cube.flux.shape[1]/2), int(self._cube.flux.shape[2]/2)

        # Load the complete fits lines log if input
        if fname is not None:
            if Path(fname).is_file():
                self.hdul_linelog = fits.open(fname, lazy_load_hdus=False)
            else:
                _logger.info(f'The lines log at {fname} was not found.')

        # Get figure configuration
        fig_conf = theme.fig_defaults(fig_cfg, fig_type='cube_interactive')
        self.axes_conf = {'image': theme.ax_defaults(ax_cfg_image, self._cube, fig_type='cube', line_bg=line_bg,
                                                     line_fg=line_fg, masks_dict=self.masks_dict, wcs=wcs),

                          'spectrum': theme.ax_defaults(ax_cfg_spec, self._cube,  g_type='default', line_bg=line_bg,
                                                        line_fg=line_fg, masks_dict=self.masks_dict, wcs=wcs)}

        grid_params = dict(nrows=1, ncols=2, width_ratios=[1, 2], height_ratios=[1])
        sub_grid_params = dict(nrows=2, ncols=1, height_ratios=[0.7, 0.3])

        # Create the figure
        with rc_context(fig_conf):

            # Figure structure
            self.fig, self.ax = plt.figure() if in_fig is None else in_fig, [None, None, None]
            gs = GridSpec(figure=self.fig, **grid_params)
            sub_gs = gs if len(self.masks_dict) == 0 else GridSpecFromSubplotSpec(subplot_spec=gs[0], **sub_grid_params)

            # Image and spectrum axes
            self.ax[0] = self.fig.add_subplot(sub_gs[0]) if wcs is None else self.fig.add_subplot(sub_gs[0],
                                                                                                 projection=wcs,
                                                                                                 slices=slices)
            self.ax[1] = self.fig.add_subplot(gs[1])
            self.ax[2] = None if len(self.masks_dict) == 0 else self.fig.add_subplot(sub_gs[1])

            # Buttons axis if provided
            if self.ax[2] is not None:
                radio = RadioButtons(self.ax[2],
                                     labels=list(self.masks_dict.keys()),
                                     radio_props={'s': [10] * len(self.masks_dict)},
                                     label_props={'fontsize': [5] * len(self.masks_dict)})
                radio.on_clicked(self.mask_selection)

            # Plot the data
            self.data_plots()

            # Connect to the toolbar
            self.toolbar = plt.get_current_fig_manager().toolbar

            # Connect the widgets
            self.fig.canvas.mpl_connect('axes_enter_event', self.on_enter_axes)
            self.fig.canvas.mpl_connect('button_press_event', self.on_click)
            self.fig.canvas.mpl_connect('button_release_event', self.click_zoom)

            # Display the figure
            save_close_fig_swicth(maximise=maximize, bbox_inches='tight', plot_check=True if in_fig is None else False)

            # Close the lines log if it has been opened
            if isinstance(self.hdul_linelog, fits.hdu.HDUList):
                self.hdul_linelog.close()

        return

    def data_plots(self, show_profiles=True):

        # Delete previous marker
        if self.marker is not None:
            self.marker.remove()
            self.marker = None

        # Background image
        self.im, _, self.marker = image_plot(self.ax[0], self.bg_image, self.fg_image, self.fg_levels, self.fg_mesh,
                                        self.bg_scale, self.fg_scale, self.bg_color, self.fg_color, self.key_coords)

        # Spatial masks
        spatial_mask_plot(self.ax[0], self.masks_dict, self.mask_color, self.mask_alpha, self._cube.units_flux,
                          mask_list=[self.mask_ext])

        self.ax[0].update(self.axes_conf['image'])

        # Voxel spectrum
        spec = self.get_spaxel_spec()
        wave_plot, flux_plot, err_plot, z_corr, idcs_mask = frame_mask_switch(spec, self.rest_frame)

        # Plot the spectrum
        self.ax[1].step(wave_plot / z_corr, flux_plot * z_corr, where='mid', color=theme.colors['fg'],
                linewidth=theme.plt['spectrum_width'])

        # Plot the fittings
        if show_profiles and spec.frame.size > 0:
            mplcursor_list = []
            for line_label in unique_line_arr(spec.frame):
                line = Line.from_transition(line_label, data_frame=spec.frame)
                mplcursor_list += spec_profile_plotter(self.ax[1], spec, line, z_corr)

            # Pop-ups
            mplcursor_parser(mplcursor_list, spec)

        # Y scale
        if self.log_scale:
            self.ax[1].set_yscale('log')

        # Update the axis
        self.axes_conf['spectrum']['title'] = f'Spaxel {self.key_coords[0]} - {self.key_coords[1]}'
        self.ax[1].update(self.axes_conf['spectrum'])

        return

    def on_click(self, event, new_voxel_button=3):

        if self.in_ax == self.ax[0]:

            # Save axes zoom
            self.save_zoom()

            if event.button == self.spaxel_button:

                # Save clicked coordinates for next plot
                self.key_coords = np.rint(event.ydata).astype(int), np.rint(event.xdata).astype(int)

                # Replot the figure
                self.im.remove()
                self.ax[1].clear()
                self.data_plots()

                self.reset_zoom()
                self.fig.canvas.draw()

            # if event.dblclick:
            if event.button == self.add_remove_button:
                if len(self.masks_dict) > 0:

                    # Save clicked coordinates for next plot
                    self.key_coords = np.rint(event.ydata).astype(int), np.rint(event.xdata).astype(int)

                    # Add or remove voxel from mask:
                    self.spaxel_selection()

                    # Save the new mask: # TODO just update the one we need
                    hdul = fits.HDUList([fits.PrimaryHDU()])
                    for mask_name, mask_attr in self.masks_dict.items():
                        hdul.append(fits.ImageHDU(name=mask_name, data=mask_attr[0].astype(int), ver=1, header=mask_attr[1]))
                    hdul.writeto(self.mask_file, overwrite=True, output_verify='fix')

                    # Replot the figure
                    self.im.remove()
                    self.ax[1].clear()
                    self.data_plots()

                    self.reset_zoom()
                    self.fig.canvas.draw()

            return

    def mask_selection(self, mask_label):

        # Assign the mask
        self.mask_ext = mask_label

        # Zoom storage
        self.save_zoom()

        # Replot the figure
        self.im.remove()
        self.ax[1].clear()
        self.data_plots()

        self.reset_zoom()
        self.fig.canvas.draw()

        return

    def spaxel_selection(self):

        for mask, mask_data in self.masks_dict.items():
            mask_matrix = mask_data[0]
            if mask == self.mask_ext:
                mask_matrix[self.key_coords[0], self.key_coords[1]] = not mask_matrix[self.key_coords[0], self.key_coords[1]]
            else:
                mask_matrix[self.key_coords[0], self.key_coords[1]] = False

            self.masks_dict[mask] = mask_data

        return

    def on_enter_axes(self, event):
        self.in_ax = event.inaxes

        return

    def save_zoom(self):
        self.axlim_dict['image_xlim'] = self.ax[0].get_xlim()
        self.axlim_dict['image_ylim'] = self.ax[0].get_ylim()
        self.axlim_dict['spec_xlim'] = self.ax[1].get_xlim()
        self.axlim_dict['spec_ylim'] = self.ax[1].get_ylim()

        return

    def reset_zoom(self):

        if self.restore_zoom:
            self.ax[0].set_xlim(self.axlim_dict['image_xlim'])
            self.ax[0].set_ylim(self.axlim_dict['image_ylim'])
            self.ax[1].set_xlim(self.axlim_dict['spec_xlim'])

            if self.maintain_y_zoom:
                self.ax[1].set_ylim(self.axlim_dict['spec_ylim'])

        else:
            self.ax[1].relim()
            self.ax[1].autoscale_view()

        return

    def click_home(self):
        self.restore_zoom = False
        self.ax[1].relim()
        self.ax[1].autoscale_view()

        return

    def click_zoom(self, event):

        if self.in_ax == self.ax[1]:
            if self.toolbar.mode == 'zoom rect' or self.toolbar.mode == 'pan/zoom':
                self.restore_zoom = True

        return

    def get_spaxel_spec(self):

        if self.key_coords is not None:
            idx_j, idx_i = self.key_coords
            spec = self._cube.get_spectrum(idx_j, idx_i)

            # Check if lines have been measured
            if self.hdul_linelog is not None:
                ext_name = f'{idx_j}-{idx_i}{self.ext_log}'

                # Better sorry than permission. Faster?
                try:
                    log = pd.DataFrame.from_records(data=self.hdul_linelog[ext_name].data, index='index')
                    spec.load_frame(log)

                except KeyError:
                    _logger.info(f'Extension {ext_name} not found in the input file')

            return spec

        else:
            return None


class SpectrumCheck(Plotter, BandsInspection):

    def __init__(self, spectrum):

        # Instantiate the dependencies
        Plotter.__init__(self)
        BandsInspection.__init__(self)

        # Lime spectrum object with the scientific data
        self._spec = spectrum

        # Variables for the matplotlib figures
        self._fig, self._ax = None, None

        return


class CubeCheck(Plotter, CubeInspection):

    def __init__(self, cube):

        # Instantiate the dependencies
        Plotter.__init__(self)
        CubeInspection.__init__(self)

        # Lime cube object with the scientific data
        self._cube = cube

        # Variables for the matplotlib figures
        self._fig, self._ax = None, None

        return


class SampleCheck(Plotter, RedshiftInspection):

    def __init__(self, sample):

        # Instantiate the dependencies
        Plotter.__init__(self)
        RedshiftInspection.__init__(self)

        # Lime spectrum object with the scientific data
        self._sample = sample

        # Variables for the matplotlib figures
        self._fig, self._ax = None, None

        return
