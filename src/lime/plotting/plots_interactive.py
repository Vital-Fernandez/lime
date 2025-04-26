import logging
import numpy as np
import pandas as pd

from pathlib import Path
from matplotlib import pyplot as plt, gridspec, rc_context
from matplotlib.widgets import RadioButtons, SpanSelector, Slider
from matplotlib.ticker import NullLocator
from astropy.io import fits
from re import sub

from ..io import load_frame, save_frame, LiMe_Error, check_file_dataframe
from .plots import Plotter, frame_mask_switch, save_close_fig_swicth, _auto_flux_scale,\
                    determine_cube_images, load_spatial_mask, check_image_size, \
                    image_plot, spec_plot, spatial_mask_plot, _masks_plot, bands_filling_plot, theme
from ..tools import pd_get

from ..transitions import label_decomposition, Line

_logger = logging.getLogger('LiMe')


def establish_selection_lines(spec, input_log, band_vsigma, n_sigma, adjust_central_band, instrumental_correction,
                              components_detection, composite_lines, fit_cfg, default_cfg_prefix, obj_cfg_prefix,
                              update_default, line_list, particle_list, decimals, vacuum_waves, ref_bands, update_labels,
                              update_latex, vacuum_label):

    # Use the default database crop for the observation if none provided
    ref_bands = spec.retrieve.line_bands(band_vsigma=band_vsigma, n_sigma=n_sigma, adjust_central_band=adjust_central_band,
                                         instrumental_correction=instrumental_correction, components_detection=False,
                                         composite_lines=None, fit_cfg=None, default_cfg_prefix=None,
                                         obj_cfg_prefix=None, update_default=True,
                                         line_list=line_list, particle_list=particle_list, decimals=decimals,
                                         vacuum_waves=vacuum_waves, ref_bands=ref_bands, update_labels=update_labels,
                                         update_latex=update_latex, vacuum_label=vacuum_label)


    # Load input log or copy the reference one
    in_bands = check_file_dataframe(input_log, verbose=False)
    if in_bands is None:
        in_bands = spec.retrieve.line_bands(band_vsigma=band_vsigma, n_sigma=n_sigma, adjust_central_band=adjust_central_band,
                                         instrumental_correction=instrumental_correction, components_detection=components_detection,
                                         composite_lines=composite_lines, fit_cfg=fit_cfg, default_cfg_prefix=default_cfg_prefix,
                                         obj_cfg_prefix=obj_cfg_prefix, update_default=update_default,
                                         line_list=line_list, particle_list=particle_list, decimals=decimals,
                                         vacuum_waves=vacuum_waves, ref_bands=ref_bands, update_labels=update_labels,
                                         update_latex=update_latex, vacuum_label=vacuum_label)
        default_status = 1 if components_detection else 0
    else:
        default_status = 1

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
        wave_arr = label_decomposition(log.index.to_numpy(), params_list=['wavelength'])
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

        self._fig = None
        self.ax_list = None
        self._y_scale = None
        self._log_address = None
        self._rest_frame = None
        self._z_slider = None
        self._z_orig = None
        self._sweep_mask = 0
        self._inter_mask = None
        self._rest_frame = True

        self.exclude_continua = None

        self.line = None
        self.mask = None
        self.log = None

        self._color_bg = {True: theme.colors['inspection_positive'],
                          False: theme.colors['inspection_negative']}

        self._out_params = ["wavelength", "wave_vac", "w1", "w2", "w3", "w4", "w5", "w6",
                            "units_wave", "particle", "transition", "rel_int"]

        return

    def bands(self, bands_file, band_vsigma=70, n_sigma=4, adjust_central_band=True, instrumental_correction=True,
              components_detection=False, composite_lines=None, fit_cfg=None, default_cfg_prefix='default',
              obj_cfg_prefix=None, update_default=True, line_list=None, particle_list=None, decimals=None,
              vacuum_waves=False, ref_bands=None, update_labels=False, update_latex=False, vacuum_label=False,
              y_scale='auto', n_cols=6, n_rows=None, col_row_scale=(1, 0.5),
              exclude_continua=False, n_pixels=10, fig_cfg=None, in_fig=None, maximize=False):


        """
        This function launches an interactive plot from which to select the line bands on the observed spectrum. If this
        function is run a second time, the user selections won't be overwritten.

        The ``bands_file`` argument provides to the output database on which the user selections will be saved.

        The ``ref_bands`` argument provides the reference database. The default database will be used if none is provided.

        The ``y_scale`` sets the flux scale for the lines grid.
        The default "auto" value automatically switches between the matplotlib `scale keywords
        <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.yscale.html>`_, otherwise the user can set a
        uniform scale for all.

        If the user wants to adjust the observation redshift and save the new values the ``z_log_address`` sets an output
        dataframe. The ``object_label`` and ``z_column`` provide the row and column indexes to save the new redshift.

        The default axes and plot titles can be modified via the ``ax_cfg``. These dictionary keys are "xlabel", "ylabel"
        and "title". It is not necessary to include all the keys in this argument.

        The `online documentation <https://lime-stable.readthedocs.io/en/latest/tutorials/n_tutorial2_lines_inspection.html>`_
        provides more details on the mechanics of this plot function.

        :param bands_file: Output file address for user bands selection.
        :type bands_file: str, pathlib.Path

        :param ref_bands: Reference bands dataframe or its file address. The default database will be used if none is provided.
        :type ref_bands: pandas.Dataframe, str, pathlib.Path, optional

        :param y_scale: Matplotlib `scale keywords <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.yscale.html>`_.
                        The default value is "auto".
        :type y_scale: str, optional.

        :param n_cols: Number of columns in plot grid. The default value is 6.
        :type n_cols: int, optional.

        :param n_rows: Number of rows in plot grid.
        :type n_rows: int, optional.

        :param col_row_scale: Multiplicative factor for the grid plots width and height. The default value is (2, 1.5).
        :type col_row_scale: tuple, optional.

        :param z_log_address: Output address for redshift dataframe file.
        :type z_log_address: str, pathlib.Path, optional

        :param object_label: Object label for redshift dataframe row indexing.
        :type object_label: str, optional

        :param z_column: Column label for redshift dataframe column indexing. The default value is "redshift".
        :type z_column: str, optional

        :param n_pixels: Maximum number of pixels for the bands correction slider. The default value is 10.
        :type n_pixels: int, optional

        :param fig_cfg: Dictionary with the matplotlib `rcParams parameters <https://matplotlib.org/stable/tutorials/introductory/customizing.html#customizing-with-dynamic-rc-settings>`_ .
        :type fig_cfg: dict, optional

        :param ax_cfg: Dictionary with the plot "xlabel", "ylabel" and "title" values.
        :type ax_cfg: dict, optional

        :param maximize: Maximise plot window. The default value is False.
        :type maximize:  bool, optional

        """

        # Assign the attribute values
        self._y_scale = y_scale
        self.exclude_continua = exclude_continua

        # Input is a mask is address and it will be also used to save the file
        if self._log_address is None:
            if isinstance(bands_file, (str, Path)):
                self._log_address = Path(bands_file)
                if not self._log_address.parent.is_dir():
                    raise LiMe_Error(f'Input bands file directory does not exist ({self._log_address.parent.as_posix()})')

        # Establish the reference lines log to inspect the mask
        self.log, self.line_list, self.active_lines = establish_selection_lines(self._spec, self._log_address,
                                                      band_vsigma, n_sigma, adjust_central_band, instrumental_correction,
                                                      components_detection, composite_lines, fit_cfg, default_cfg_prefix,
                                                      obj_cfg_prefix, update_default, line_list, particle_list, decimals,
                                                      vacuum_waves, ref_bands, update_labels, update_latex, vacuum_label)

        # Proceed if there are lines in the mask for the object spectrum wavelength range
        if len(self.log.index) > 0:

            # Establish the initial list of lines
            n_lines = self.log.index.size

            # Compute the number of rows configuration
            if n_lines > n_cols:
                if n_rows is None:
                    n_rows = int(np.ceil(n_lines / n_cols))
            else:
                n_cols, n_rows = n_lines, 1
            n_grid = n_cols * n_rows

            # Set the plot format where the user's overwrites the default
            size_conf = {'figure.figsize': (n_cols * col_row_scale[0], n_rows * col_row_scale[1])}
            size_conf = size_conf if fig_cfg is None else {**size_conf, **fig_cfg}

            PLT_CONF = theme.fig_defaults(size_conf, fig_type='grid')

            # Launch the interative figure
            with rc_context(PLT_CONF):

                # Main structure
                self._fig = plt.figure(layout='tight') if in_fig is None else in_fig
                gs0 = self._fig.add_gridspec(2, 1, height_ratios=[1, 0.1])
                gs_lines = gs0[0].subgridspec(n_rows, n_cols, hspace=0.5)

                self.ax_list = gs_lines.subplots().flatten() if n_lines > 1 else [gs_lines.subplots()]

                # Generate plot
                spanSelectDict = {}
                for i in range(n_grid):
                    if i < n_lines:
                        self.line = self.line_list[i] # log.sort_values(by=['wavelength'], inplace=True)#Line(self._lineList[i], band=self.log)
                        self._plot_line_BI(self.ax_list[i], self.line, self._rest_frame, self._y_scale)
                        spanSelectDict[f'spanner_{i}'] = SpanSelector(self.ax_list[i],
                                                                      self._on_select_MI,
                                                                      'horizontal',
                                                                      useblit=True,
                                                                      props=dict(alpha=0.5, facecolor='tab:blue'),
                                                                      button=1)
                    else:
                        # Clear not filled axes
                        self._fig.delaxes(self.ax_list[i])

                # Create sliders grid
                gs_slicers = gs0[1].subgridspec(1, 1)
                ax_sliders = [gs_slicers.subplots()]

                # Mask sweep slider configuration
                self._inter_mask = np.median(np.diff(self._spec.wave_rest.data))
                mask_slider = Slider(ax_sliders[0], 'Band\n(pixels)', -n_pixels, n_pixels, valinit=0.0, valstep=1)
                mask_slider.on_changed(self._on_mask_slider_MI)

                # Connecting the figure to the interactive widgets
                self._fig.canvas.mpl_connect('button_press_event', self._on_click_MI)
                self._fig.canvas.mpl_connect('axes_enter_event', self._on_enter_axes_MI)

                # Show the image
                save_close_fig_swicth(None, None, self._fig, maximise=maximize,
                                      plot_check=True if in_fig is None else False)

        else:
            _logger.warning(f'No lines found in the lines mask for the object wavelentgh range')

        return

    def _plot_line_BI(self, ax, line, frame, y_scale='auto', scale_dict=theme.plt):


        # Reference _frame for the plot
        wave_plot, flux_plot, z_corr, idcs_mask = frame_mask_switch(self._spec.wave, self._spec.flux,
                                                                    self._spec.redshift, frame)

        # Establish the limits for the line spectrum plot
        mask = self.log.loc[line, 'w1':'w6'].to_numpy() * z_corr
        idcsM = np.searchsorted(wave_plot, mask)

        # Just the center region is adjusted
        if self.exclude_continua:
            idxL = idcsM[2] - 10 if idcsM[2] - 10 > 0 else 0
            idxH = idcsM[3] + 10 if idcsM[3] + 10 < wave_plot.size - 1 else wave_plot.size - 1

        # Center + continua
        else:
            idxL = idcsM[0] - 5 if idcsM[0] - 5 > 0 else 0
            idxH = idcsM[5] + 5 if idcsM[5] + 5 < wave_plot.size - 1 else  wave_plot.size - 1

        # Plot the spectrum
        ax.step(wave_plot[idxL:idxH]/z_corr, flux_plot[idxL:idxH]*z_corr, where='mid', color=theme.colors['fg'],
                linewidth=scale_dict['spectrum_width'])

        # Continuum bands
        bands_filling_plot(ax, wave_plot, flux_plot, z_corr, idcsM, line, self.exclude_continua, theme.colors)

        # Plot the masked pixels
        _masks_plot(ax, [line], wave_plot[idxL:idxH], flux_plot[idxL:idxH], z_corr, self.log, idcs_mask[idxL:idxH],
                    color_dict=theme.colors)

        # Plot line location
        wave_line = pd_get(self.log, line, 'wavelength')
        if wave_line is not None:
            ax.axvline(wave_line, linestyle='--', color='grey', linewidth=0.5)

        # Background for selective line for selected lines
        ax.set_facecolor(self._color_bg[self.active_lines[self.line_list == line][0]])

        # Scale each
        _auto_flux_scale(ax, flux_plot[idxL:idxH]*z_corr, y_scale)

        # Formatting the figure
        ax.set_title(line, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.yaxis.set_minor_locator(NullLocator())
        ax.get_xlim() # TODO without this one there is no plot

        return

    def _on_select_MI(self, w_low, w_high):

        # Check we are not just clicking on the plot
        if w_low != w_high:

            # Just the central bands             # TODO Add JUSTCENTER new action maybe just move the continua bands automatically
            if self.exclude_continua:
                self.log.at[self.line, 'w3'] = w_low
                self.log.at[self.line, 'w4'] = w_high

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
            save_or_clear_log(self.log, self._log_address, self.active_lines, self._out_params)

            # Redraw the line measurement
            self._ax.clear()
            self._plot_line_BI(self._ax, self.line, frame=self._rest_frame, y_scale=self._y_scale)
            self._fig.canvas.draw()

        return

    def _on_enter_axes_MI(self, event):

        # Assign current line and axis
        self._ax = event.inaxes
        title = self._ax.get_title()
        if title != '':
            self.line = title

    def _on_click_MI(self, event):

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
            save_or_clear_log(self.log, self._log_address, self.active_lines, self._out_params)

            # Plot the line selection with the new Background
            self._ax.clear()
            self._plot_line_BI(self._ax, self.line, self._rest_frame, self._y_scale)
            self._fig.canvas.draw()

        return

    def _on_z_slider_MI(self, val):

        # Update the redshift value
        z_new = self._z_orig + (self._inter_z * val)
        self._spec.update_redshift(z_new)

        # Re-plot the lines
        for i, line in enumerate(self._lineList):
            self.ax_list[i].clear()
            self._plot_line_BI(self.ax_list[i], line, self._rest_frame, self._y_scale)
        self._fig.canvas.draw()

        # Save to the data frame
        self.z_df.loc[self._obj_ref, self._redshift_column] = z_new
        save_frame(self._redshift_log_path, self.z_df, )

        return

    def _on_mask_slider_MI(self, val):

        # Remove the previous value
        mask_matrix = self.log.loc[:, 'w1':'w6'].to_numpy() - self._sweep_mask
        self._sweep_mask = val * self._inter_mask

        # Add new sweep
        mask_matrix = mask_matrix + self._sweep_mask
        self.log.loc[:, 'w1':'w6'] = mask_matrix

        # Replot the lines
        for i, line in enumerate(self.line_list):
            print(line, self.log.loc[line, 'w1':'w6'].to_numpy())
            self.ax_list[i].clear()
            self._plot_line_BI(self.ax_list[i], line, self._rest_frame, self._y_scale)

        self._fig.canvas.draw()

        # Save the log to the file
        save_or_clear_log(self.log, self._log_address, self.active_lines, self._out_params)

        return


class RedshiftInspectionSingle:

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

    def redshift(self, reference_lines, output_file_log=None, output_idcs=None, redshift_column='redshift',
                 none_value=np.nan, unknown_value=0.0, legend_handle='levels', maximize=False, title_label=None,
                 output_address=None, fig_cfg={}, ax_cfg={}, in_fig=None):

        # Assign the attributes
        self._obj_idcs = None
        self._output_idcs = None
        self._column_log = redshift_column
        self._none_value = none_value
        self._unknown_value = unknown_value
        self._spec_label = "" if title_label is None else title_label
        self._legend_handle = legend_handle
        self._user_point = None

        # Read initial value for the redshift
        redshift_pred = self._spec.redshift
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
        self._AXES_CONF = theme.ax_defaults(ax_cfg, self._spec.units_wave, self._spec.units_flux, self._spec.norm_flux, fig_type=None)

        # Create and fill the figure
        with rc_context(PLT_CONF):

            # Generate the figure object and figures
            self._fig = plt.figure() if in_fig is None else in_fig
            gs = gridspec.GridSpec(nrows=1, ncols=2, figure=self._fig, width_ratios=[2, 0.5], height_ratios=[1])
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
            save_close_fig_swicth(output_address, 'tight', self._fig, maximise=maximize)

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

            # Load the spectrum
            spec = self._sample.load_function(self._sample.frame, obj_idx, **self._sample.load_params)

            # Plot on the observed frame with reshift = 0
            wave_plot, flux_plot, z_corr, idcs_mask = frame_mask_switch(spec.wave, spec.flux, 0, 'observed')

            # Plot the spectrum
            ax.step(wave_plot/z_corr, flux_plot*z_corr, label=self._label_generator(obj_idx), where='mid',
                    linewidth=theme.colors['spectrum_width'])

            # Plot the masked pixels
            _masks_plot(ax, None, wave_plot, flux_plot, z_corr, spec.frame, idcs_mask, color_dict=self._color_dict)

        return

    def _plot_line_labels_ZI(self, ax, click_coord, redshift_pred):

        if (redshift_pred != 0) and (not pd.isnull(redshift_pred)):
            wave_min, wave_max = None, None
            for obj_idx in self._obj_idcs:

                # Load the spectrum
                spec = self._sample.load_function(self._sample.frame, obj_idx, **self._sample.load_params)

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
                    color_line = self._color_dict['fg']

                ax.axvline(x=lineWave * (1 + redshift_pred), color=color_line, linestyle='--', linewidth=0.5)

                ax.annotate(latexRange[i], xy=(lineWave * (1 + redshift_pred), 0.85),
                            horizontalalignment="center",
                            rotation=90,
                            backgroundcolor='w',
                            size=6,
                            xycoords='data', xytext=(lineWave * (1 + redshift_pred), 0.85), textcoords=("data",
                                                                                                        "axes fraction"))

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
        self._AXES_CONF = theme.ax_defaults(ax_cfg, self._sample.load_params.get('units_wave'),
                                            self._sample.load_params.get('units_flux'),
                                            self._sample.load_params.get('norm_flux'), fig_type=None)

        # Create and fill the figure
        with rc_context(PLT_CONF):

            # Generate the figure object and figures
            self._fig = plt.figure() if in_fig is None else in_fig
            gs = gridspec.GridSpec(nrows=1, ncols=2, figure=self._fig, width_ratios=[2, 0.5], height_ratios=[1])
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
            save_close_fig_swicth(output_address, 'tight', self._fig, maximise=maximize)

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
            spec = self._sample.load_function(self._sample.frame, obj_idx, self._sample.file_address, **self._load_params)

            # Plot on the observed frame with reshift = 0
            wave_plot, flux_plot, z_corr, idcs_mask = frame_mask_switch(spec.wave, spec.flux, spec.redshift, 'observed')

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
                spec = self._sample.load_function(self._sample.frame, obj_idx, self._sample.file_address, **self._load_params)

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

        # Mask correction attributes
        self.mask_file = None
        self.mask_ext = None
        self.masks_dict = {}
        self.mask_color = None
        self.mask_array = None

        # Plot attributes
        self._ax0, self._ax1, self._ax2, self.in_ax = None, None, None, None
        self.fig_conf = None
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

    def cube(self, line, bands=None, line_fg=None, min_pctl_bg=60, cont_pctls_fg=(90, 95, 99), bg_cmap='gray',
             fg_cmap='viridis', bg_norm=None, fg_norm=None, masks_file=None, masks_cmap='viridis_r', masks_alpha=0.2,
             rest_frame=False, log_scale=False, fig_cfg=None, ax_cfg_image=None, ax_cfg_spec=None, in_fig=None,
             lines_file=None, ext_frame_suffix='_LINELOG', maintain_y_zoom=True, wcs=None, spaxel_selection_button=1,
             add_remove_button=3, maximize=False):

        """

        This function opens an interactive plot to display and individual spaxel spectrum from the selection on the
        image map.

        The left-hand side plot displays an image map with the flux sum of a line band as described on the
        Cube.plot.cube documentation.

        A right-click on a spaxel of the band image map will plot the selected spaxel spectrum on the right-hand side plot.
        This will also mark the spaxel with a red cross.

        If the user provides a ``masks_file`` the plot window will include a dot mask selector. Activating one mask will
        overplotted on the image band. A middle button click on the image band will add/remove a spaxel to the current
        pixel selected masks. If the spaxel was part of another mask it will be removed from the previous mask region.

        If the user provides a ``lines_log_file`` .fits file, the fitted profiles will be included on its corresponding
        spaxel spectrum plot. The measurements logs on this ".fits" file must be named using the spaxel array coordinate
        and the suffix on the ``ext_log`` argument.

        If the user has installed the library `mplcursors <https://mplcursors.readthedocs.io/en/stable/>`_, a left-click
        on a fitted profile will pop-up properties of the fitting, right-click to delete the annotation.

        By default the left mouse button selects the displayed spaxel (if you use the zoom tool the first click
        will switch the spaxel). The right mouse button will add/remove pixels from the current mask. You can switch the
        mouse buttons for these operations using the Matplotlib classification: LEFT = 1, MIDDLE = 2, RIGHT = 3, BACK = 8
        and FORWARD = 9


        :param line: Line label for the spatial map background image.
        :type line: str

        :param bands: Bands dataframe (or file address to the dataframe).
        :type bands: pandas.Dataframe, str, path.Pathlib, optional

        :param line_fg: Line label for the spatial map background image contours
        :type line_fg: str, optional

        :param min_pctl_bg: Minimum band flux percentile for spatial map background image. The default value is 60.
        :type min_pctl_bg: float, optional

        :param cont_pctls_fg: Band flux percentiles for foreground ``line_fg`` contours. The default value is (90, 95, 99)
        :type cont_pctls_fg: tuple, optional

        :param bg_cmap: Background image flux `color map <https://matplotlib.org/stable/gallery/images_contours_and_fields/colormap_normalizations.html>`_.
                        The default value is "gray".
        :type bg_cmap: str, optional

        :param fg_cmap: Foreground image flux `color map <https://matplotlib.org/stable/gallery/images_contours_and_fields/colormap_normalizations.html>`_.
                        The default value is "viridis".
        :type fg_cmap: str, optional

        :param bg_norm: Background image `color normalization <https://matplotlib.org/stable/gallery/images_contours_and_fields/colormap_normalizations.html>`_.
                        The default value is `SymLogNorm <https://matplotlib.org/stable/gallery/images_contours_and_fields/colormap_normalizations.html>`_.
        :type bg_norm: Normalization from matplotlib.colors, optional

        :param fg_norm: Foreground contours `color normalization <https://matplotlib.org/stable/gallery/images_contours_and_fields/colormap_normalizations.html>`_.
                        The default value is `LogNorm <https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.LogNorm.html>`_.
        :type fg_norm: Normalization from matplotlib.colors, optional

        :param masks_file: File address for binary spatial masks
        :type masks_file: str, optional

        :param masks_cmap: Binary masks `color map <https://matplotlib.org/stable/gallery/images_contours_and_fields/colormap_normalizations.html>`_.
        :type masks_cmap: str, optional

        :param masks_alpha: Transparency alpha value. The default value is 0.2 (0 to 1 scale).
        :type masks_alpha: float, optional

        :param rest_frame: Set to True for a display in rest frame. The default value is False
        :type rest_frame: bool, optional

        :param log_scale: Set to True for a display with a logarithmic scale flux. The default value is False
        :type log_scale: bool, optional

        :param fig_cfg: `Matplotlib RcParams <https://matplotlib.org/stable/api/matplotlib_configuration_api.html#matplotlib.RcParams>`_
                        parameters for the figure format
        :type fig_cfg: dict, optional

        :param ax_cfg_image: Dictionary with the image band flux plot "xlabel", "ylabel" and "title" values.
        :type ax_cfg_image: dict, optional

        :param ax_cfg_spec: Dictionary with the spaxel spectrum plot "xlabel", "ylabel" and "title" values.
        :type ax_cfg_spec: dict, optional

        :param lines_file: Address for the line measurements log ".fits" file.
        :type lines_file: str, pathlib.Path, optional

        :param ext_frame_suffix: Suffix for the line measurements log spaxel page name
        :type ext_frame_suffix: str, optional

        :param wcs: Observation `world coordinate system <https://docs.astropy.org/en/stable/wcs/index.html>`_.
        :type wcs: astropy WCS, optional

        :param spaxel_selection_button: Mouse button for switching displayed spaxel. The default value is 1 (left button).
        :type spaxel_selection_button: int, optional

        :param add_remove_button: Mouse button to add/remove spaxels from the current mask. The default value is 2 (right button).
        :type add_remove_button: int, optional

        :param maximize: Maximise plot window. The default value is False.
        :type maximize:  bool, optional


        """

        self.ext_log = ext_frame_suffix
        self.mask_file = masks_file
        self.spaxel_button = spaxel_selection_button
        self.add_remove_button = add_remove_button
        self.maintain_y_zoom = maintain_y_zoom

        # Prepare the background image data
        line_bg, self.bg_image, self.bg_levels, self.bg_scale = determine_cube_images(self._cube, line, bands,
                                                                                      min_pctl_bg, bg_norm, contours_check=False)

        # Prepare the foreground image data
        line_fg, self.fg_image, self.fg_levels, self.fg_scale = determine_cube_images(self._cube, line_fg, bands,
                                                                                      cont_pctls_fg, fg_norm, contours_check=True)

        # Mesh for the countours
        if line_fg is not None:
            y, x = np.arange(0, self.fg_image.shape[0]), np.arange(0, self.fg_image.shape[1])
            self.fg_mesh = np.meshgrid(x, y)
        else:
            self.fg_mesh = None

        # Colors
        self.bg_color, self.fg_color, self.mask_color, self.mask_alpha = bg_cmap, fg_cmap, masks_cmap, masks_alpha

        # Frame
        self.rest_frame, self.log_scale = rest_frame, log_scale

        # Load the masks
        self.masks_dict = load_spatial_mask(self.mask_file)

        # Check that the images have the same size
        check_image_size(self.bg_image, self.fg_image, self.masks_dict)

        # Use the input wcs or use the parent one
        wcs = self._cube.wcs if wcs is None else wcs

        # Image mesh grid
        frame_size = self._cube.flux.shape
        y, x = np.arange(0, frame_size[1]), np.arange(0, frame_size[2])
        self.grid_mesh = np.meshgrid(x, y)

        # Use central voxel as initial coordinate
        self.key_coords = int(self._cube.flux.shape[1]/2), int(self._cube.flux.shape[2]/2)

        if len(self.masks_dict) > 0:
            self.mask_ext = list(self.masks_dict.keys())[0]
        else:
            self.mask_ext = self.ext_log

        # Load the complete fits lines log if input
        if lines_file is not None:
            if Path(lines_file).is_file():
                self.hdul_linelog = fits.open(lines_file, lazy_load_hdus=False)
            else:
                _logger.info(f'The lines log at {lines_file} was not found.')

        # Get figure configuration configuration
        self.fig_conf = theme.fig_defaults(fig_cfg, fig_type='cube_interactive')
        ax_im = theme.ax_defaults(ax_cfg_image, self._cube.units_wave, self._cube.units_flux, self._cube.norm_flux,
                                fig_type='cube', line_bg=line_bg, line_fg=line_fg, masks_dict=self.masks_dict, wcs=wcs)
        ax_spec = theme.ax_defaults(ax_cfg_spec, self._cube.units_wave, self._cube.units_flux, self._cube.norm_flux,
                                g_type='default', line_bg=line_bg, line_fg=line_fg, masks_dict=self.masks_dict, wcs=wcs)
        self.axes_conf = {'image': ax_im, 'spectrum': ax_spec}

        # Create the figure
        with rc_context(self.fig_conf):

            # Figure structure
            self._fig = plt.figure() if in_fig is None else in_fig
            gs = gridspec.GridSpec(nrows=1, ncols=2, figure=self._fig, width_ratios=[1, 2], height_ratios=[1])

            # Create subgrid for buttons if mask file provided
            if len(self.masks_dict) > 0:
                gs_image = gridspec.GridSpecFromSubplotSpec(nrows=2, ncols=1, subplot_spec=gs[0], height_ratios=[0.7, 0.3])
            else:
                gs_image = gs

            # Image axes Astronomical coordinates if provided
            if wcs is None:
                self._ax0 = self._fig.add_subplot(gs_image[0])
            else:
                slices = ('x', 'y', 1) if wcs.naxis == 3 else ('x', 'y')
                self._ax0 = self._fig.add_subplot(gs_image[0], projection=wcs, slices=slices)

            # Spectrum plot
            self._ax1 = self._fig.add_subplot(gs[1])

            # Buttons axis if provided
            if len(self.masks_dict) > 0:
                self._ax2 = self._fig.add_subplot(gs_image[1])

                # Create the RadioButtons widget with the specified properties
                labels_buttons = list(self.masks_dict.keys())
                radio_props = {'s': [10] * len(labels_buttons)}
                label_props = {'fontsize': [5] * len(labels_buttons)}
                radio = RadioButtons(self._ax2, labels_buttons, radio_props=radio_props, label_props=label_props)

            # Plot the data
            self.data_plots()

            # Connect to the toolbar
            self.toolbar = plt.get_current_fig_manager().toolbar

            # For pytests on figure
            # if self.toolbar is not None:
            #     self.toolbar._actions['home'].triggered.connect(self.click_home)

            # Connect the widgets
            self._fig.canvas.mpl_connect('axes_enter_event', self.on_enter_axes)
            self._fig.canvas.mpl_connect('button_press_event', self.on_click)
            self._fig.canvas.mpl_connect('button_release_event', self.click_zoom)

            if len(self.masks_dict) > 0:
                radio.on_clicked(self.mask_selection)

            # Display the figure
            save_close_fig_swicth(maximise=maximize, bbox_inches='tight', plot_check=True if in_fig is None else False)

            # Close the lines log if it has been opened
            if isinstance(self.hdul_linelog, fits.hdu.HDUList):
                self.hdul_linelog.close()

        return

    def data_plots(self):

        # Delete previous marker
        if self.marker is not None:
            self.marker.remove()
            self.marker = None

        # Background image
        self.im, _, self.marker = image_plot(self._ax0, self.bg_image, self.fg_image, self.fg_levels, self.fg_mesh,
                                        self.bg_scale, self.fg_scale, self.bg_color, self.fg_color, self.key_coords)

        # Spatial masks
        spatial_mask_plot(self._ax0, self.masks_dict, self.mask_color, self.mask_alpha, self._cube.units_flux,
                          mask_list=[self.mask_ext])

        # Voxel spectrum
        if self.key_coords is not None:
            idx_j, idx_i = self.key_coords
            flux_voxel = self._cube.flux[:, idx_j, idx_i]
            log = None

            # Check if lines have been measured
            if self.hdul_linelog is not None:
                ext_name = f'{idx_j}-{idx_i}{self.ext_log}'

                # Better sorry than permission. Faster?
                try:
                    log = pd.DataFrame.from_records(data=self.hdul_linelog[ext_name].data, index='index')

                except KeyError:
                    _logger.info(f'Extension {ext_name} not found in the input file')

            # Plot spectrum
            spec_plot(self._ax1, self._cube.wave, flux_voxel, self._cube.redshift, self._cube.norm_flux,
                      rest_frame=self.rest_frame, log=log, units_wave=self._cube.units_wave,
                      units_flux=self._cube.units_flux)

            if self.log_scale:
                self._ax.set_yscale('log')

        # Update the axis
        self.axes_conf['spectrum']['title'] = f'Spaxel {idx_j} - {idx_i}'
        self._ax0.update(self.axes_conf['image'])
        self._ax1.update(self.axes_conf['spectrum'])

        return

    def on_click(self, event, new_voxel_button=3):

        if self.in_ax == self._ax0:

            # Save axes zoom
            self.save_zoom()

            if event.button == self.spaxel_button:

                # Save clicked coordinates for next plot
                self.key_coords = np.rint(event.ydata).astype(int), np.rint(event.xdata).astype(int)

                # Remake the drawing
                self.im.remove()# self.ax0.clear()
                self._ax1.clear()
                self.data_plots()

                # Reset the image
                self.reset_zoom()
                self._fig.canvas.draw()

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

                    # Remake the drawing
                    self.im.remove()
                    self._ax1.clear()
                    self.data_plots()

                    # Reset the image
                    self.reset_zoom()
                    self._fig.canvas.draw()

            return

    def mask_selection(self, mask_label):

        # Assign the mask
        self.mask_ext = mask_label

        # Replot the figure
        self.save_zoom()
        self.im.remove()
        self._ax1.clear()
        self.data_plots()

        # Reset the image
        self.reset_zoom()
        self._fig.canvas.draw()

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
        self.axlim_dict['image_xlim'] = self._ax0.get_xlim()
        self.axlim_dict['image_ylim'] = self._ax0.get_ylim()
        self.axlim_dict['spec_xlim'] = self._ax1.get_xlim()
        self.axlim_dict['spec_ylim'] = self._ax1.get_ylim()

        return

    def reset_zoom(self):

        if self.restore_zoom:
            self._ax0.set_xlim(self.axlim_dict['image_xlim'])
            self._ax0.set_ylim(self.axlim_dict['image_ylim'])
            self._ax1.set_xlim(self.axlim_dict['spec_xlim'])

            if not self.maintain_y_zoom:
                self._ax1.set_ylim(self.axlim_dict['spec_ylim'])

        else:
            self._ax1.relim()
            self._ax1.autoscale_view()

        return

    def click_home(self):
        self.restore_zoom = False
        self._ax1.relim()
        self._ax1.autoscale_view()

        return

    def click_zoom(self, event):

        if self.in_ax == self._ax1:
            if self.toolbar.mode == 'zoom rect' or self.toolbar.mode == 'pan/zoom':
                self.restore_zoom = True

        return


class MaskInspection:

    def __init__(self):

        self._SN_level = None


        return

    def spatial_mask(self, line, bands=None, min_pctl_bg=60, cont_pctls_fg=(90, 95, 99), bg_cmap='gray',
             fg_cmap='viridis', bg_norm=None, fg_norm=None, masks_file=None, masks_cmap='viridis_r', masks_alpha=0.2,
             fig_cfg={}, ax_cfg_image={}, ax_cfg_spec={}, in_fig=None, wcs=None, maximize=False):

        # --------------- Compute the cumilative signal to noise ---------------------------
        # TODO finish this one
        line_bg = Line(line, bands)

        # Get the band indexes
        line_bg.index_bands(self.wave, self.redshift)
        idcsEmis, idcsCont = line_bg.index_bands(self._cube.wave, self._cube.wave)
        signal_slice = self._cube.flux[idcsEmis, :, :]

        # Compute the Signal to noise in the complete image
        n_pixels = np.sum(idcsCont)
        cont_slice = self._cube.flux[idcsCont, :, :]
        Amp_image = np.nanmax(signal_slice, axis=0) - np.nanmean(cont_slice, axis=0)
        std_image = np.nanstd(cont_slice, axis=0)
        param_image = (np.sqrt(2 * n_pixels * np.pi) / 6) * (Amp_image / std_image)

        # Set nan entries == 0
        param_image[np.isnan(param_image)] = 0

        # Get the coordinates of all cells using np.argwhere()
        coordinates = np.argwhere(param_image)

        # Sort the coordinates based on the values in descending order
        sorted_coordinates = coordinates[np.argsort(param_image[coordinates[:, 0], coordinates[:, 1]])[::-1]]

        # Extract cell values based on sorted coordinates and create a 1-dimensional array
        self.sorted_values = param_image[sorted_coordinates[:, 0], sorted_coordinates[:, 1]]

        # Calculate the cumulative sum of the sorted values
        self.cumulative_sum = np.nancumsum(self.sorted_values)

        # --------------- Compute the cummulative signal to noise ---------------------------

        # Prepare the background image data
        line_bg, self.bg_image, self.bg_levels, self.bg_scale = determine_cube_images(self._cube, line, bands,
                                                                                      min_pctl_bg, bg_norm, contours_check=False)

        # Colors
        self.bg_color, self.fg_color, self.mask_color, self.mask_alpha = bg_cmap, fg_cmap, masks_cmap, masks_alpha

        # Image mesh grid
        frame_size = self._cube.flux.shape
        y, x = np.arange(0, frame_size[1]), np.arange(0, frame_size[2])
        self.grid_mesh = np.meshgrid(x, y)

        self.mask_ext = 'CUM_SN'

        # Get figure configuration configuration
        self.fig_conf = theme.fig_defaults(fig_cfg, fig_type='cube_interactive')
        ax_im = theme.ax_defaults(ax_cfg_image, self._cube.units_wave, self._cube.units_flux, self._cube.norm_flux,
                                fig_type='cube', line_bg=line_bg, line_fg=None, masks_dict=self.masks_dict, wcs=wcs)
        ax_spec = theme.ax_defaults(ax_cfg_spec, self._cube.units_wave, self._cube.units_flux, self._cube.norm_flux,
                               g_type='default', line_bg=line_bg, line_fg=None, masks_dict=self.masks_dict, wcs=wcs)
        self.axes_conf = {'image': ax_im, 'spectrum': ax_spec}

        # Create the figure
        with rc_context(self.fig_conf):

            # Figure structure
            self._fig = plt.figure() if in_fig is None else in_fig
            gs = gridspec.GridSpec(nrows=1, ncols=2, figure=self._fig, width_ratios=[1, 2], height_ratios=[1])

            # Create subgrid for buttons if mask file provided
            if len(self.masks_dict) > 0:
                gs_image = gridspec.GridSpecFromSubplotSpec(nrows=2, ncols=1, subplot_spec=gs[0], height_ratios=[0.8, 0.2])
            else:
                gs_image = gs

            # Image axes Astronomical coordinates if provided
            if wcs is None:
                self._ax0 = self._fig.add_subplot(gs_image[0])
            else:
                slices = ('x', 'y', 1) if wcs.naxis == 3 else ('x', 'y')
                self._ax0 = self._fig.add_subplot(gs_image[0], projection=wcs, slices=slices)

            # Spectrum plot
            self._ax1 = self._fig.add_subplot(gs[1])

            # Buttons axis if provided
            if len(self.masks_dict) > 0:
                self._ax2 = self._fig.add_subplot(gs_image[1])
                radio = RadioButtons(self._ax2, ['Save'])

                for r in radio.labels:
                    r.set_fontsize(10)

                for circle in radio.circles:
                    circle.set_width(0.04)

            # Plot the data
            self.data_plots()

            # Connect the widgets
            self._fig.canvas.mpl_connect('button_press_event', self.on_click)
            self._fig.canvas.mpl_connect('axes_enter_event', self.on_enter_axes)
            if len(self.masks_dict) > 0:
                radio.on_clicked(self.mask_selection)

            # Display the figure
            save_close_fig_swicth(maximise=maximize, bbox_inches='tight', plot_check=True if in_fig is None else False)


        # # Create a plot
        # plt.plot(range(1, len(cumulative_sum) + 1), cumulative_sum, marker='o', linestyle='-')
        #
        # # Add labels and title
        # plt.xlabel('Number of Cells (in descending order)')
        # plt.ylabel('Cumulative Sum of Cell Values')
        # plt.title('Cumulative Sum of Sorted Matrix Cell Values')
        #
        # # Show the plot
        # plt.grid()
        # plt.show()

        return

    def data_plots(self):

        # Delete previous marker
        if self.marker is not None:
            self.marker.remove()
            self.marker = None

        # Background image
        self.im, _, self.marker = image_plot(self._ax0, self.bg_image, self.fg_image, self.fg_levels, self.fg_mesh,
                                        self.bg_scale, self.fg_scale, self.bg_color, self.fg_color, self.key_coords)

        # Spatial masks
        # spatial_mask_plot(self._ax0, self.masks_dict, self.mask_color, self.mask_alpha, self._cube.units_flux,
        #                   mask_list=[self.mask_ext])

        # Voxel spectrum
        # if self.key_coords is not None:
        #     idx_j, idx_i = self.key_coords
        #     flux_voxel = self._cube.flux[:, idx_j, idx_i]
        #     log = None
        #
        #     # Check if lines have been measured
        #     if self.hdul_linelog is not None:
        #         ext_name = f'{idx_j}-{idx_i}{self.ext_log}'
        #
        #         # Better sorry than permission. Faster?
        #         try:
        #             log = Table.read(self.hdul_linelog[ext_name]).to_pandas()
        #             log.set_index('index', inplace=True)
        #
        #             # Change 'nan' to np.nan
        #             idcs_nan_str = log['group_label'] == 'nan'
        #             log.loc[idcs_nan_str, 'group_label'] = np.nan
        #
        #         except KeyError:
        #             _logger.info(f'Extension {ext_name} not found in the input file')
        #
        #     # Plot spectrum
        #     spec_plot(self._ax1, self._cube.wave, flux_voxel, self._cube.redshift, self._cube.norm_flux,
        #               rest_frame=self.rest_frame, log=log, units_wave=self._cube.units_wave,
        #               units_flux=self._cube.units_flux, color_dict=self._color_dict)
        #
        #     if self.log_scale:
        #         self._ax.set_yscale('log')

        # Update the axis
        # self.axes_conf['spectrum']['title'] = f'Spaxel {idx_j} - {idx_i}'
        self._ax0.update(self.axes_conf['image'])
        self._ax1.update(self.axes_conf['spectrum'])

        return


class SpectrumCheck(Plotter, RedshiftInspectionSingle, BandsInspection):

    def __init__(self, spectrum):

        # Instantiate the dependencies
        Plotter.__init__(self)
        RedshiftInspectionSingle.__init__(self)
        BandsInspection.__init__(self)

        # Lime spectrum object with the scientific data
        self._spec = spectrum

        # Variables for the matplotlib figures
        self._fig, self._ax = None, None

        return


class CubeCheck(Plotter, CubeInspection, MaskInspection):

    def __init__(self, cube):

        # Instantiate the dependencies
        Plotter.__init__(self)
        CubeInspection.__init__(self)
        MaskInspection.__init__(self)

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
