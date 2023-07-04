LiMe: A Line Measuring library
================================

.. image:: ./_static/logo_transparent.png
    :scale: 12%
    :align: center

This library provides a set of tools to fit lines in astronomical spectra. Its design aims for a user-friendly workflow
for both single lines and large data sets. The library provides tools for masking, detecting, profile fitting
and storing the results. The output measurements are focused on the gas chemical and kinematic analysis.

These are some of the features currently available:

* Non-profile and Gaussian profile emission and absorption line measurements.
* The user can include the pixel error spectrum in the calculation.
* The Multi-Gaussian profile parameters can be constrained by the user during the fitting.
* Tools to confirm the presence of lines.
* Static and interactive plots for the visual appraisal of inputs and outputs
* Line labels adhere to the `PyNeb <http://research.iac.es/proyecto/PyNeb/>`_ format.
* The measurements can be saved in several file types, including multi-page *.fits*, *.asdf* and *.xlsx* files

.. admonition:: Where to find what you need
   :class: hint

   💻 To install or update the library go to the `installation page <documentation/installation.html>`_. Download the
   `sample data folder <https://github.com/Vital-Fernandez/lime/tree/master/examples/sample_data>`_ and try to run the
   `examples <https://github.com/Vital-Fernandez/lime/tree/master/examples>`_.

   🚀 For a quick start go to the **Tutorials** section. These examples are organized by increasing complexity and they
   provide a working knowledge of the library algorithms.

   🌀 To learn more about the library design, please check the **Inputs** section to understand how to adapt LiMe to your workflow.
   The library functions description can be found at the :ref:`API <api>`.

   📈 The tabulated and graphical measurements description is available in the **Outputs** section.


.. :ref:`doc-tree`

.. toctree::
   :maxdepth: 1
   :caption: Reference
   :name: doc-tree

   introduction/installation
   introduction/api

.. toctree::
   :maxdepth: 1
   :caption: Inputs
   :name: input-tree

   inputs/n_inputs1_spectra.ipynb
   inputs/n_inputs2_line_labels.ipynb
   inputs/n_inputs3_line_bands.ipynb
   inputs/n_inputs4_fit_configuration.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Outputs
   :name: output-tree

   outputs/outputs1_measurements.rst
   outputs/n_outputs2_plots.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Tutorials
   :name: tutorial-tree

   tutorials/n_tutorial1_single_line_fit.ipynb
   tutorials/n_tutorial2_lines_inspection.ipynb
   tutorials/n_tutorial3_complete_spectrum.ipynb
   tutorials/n_tutorial4_IFU_masking.ipynb
   tutorials/n_tutorial5_IFU_fitting.ipynb
   tutorials/n_tutorial6_IFU_results.ipynb
