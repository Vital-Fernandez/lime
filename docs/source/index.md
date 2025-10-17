# LiMe: A Line Measuring library (v2 release)

```{image} 0_resources/images/LiMe2_logo_white_transparent.png
:width: 500px
:align: center
:class: only-light
```

```{image} 0_resources/images/LiMe2_logo_dark_transparent.png
:width: 500px
:align: center
:class: only-dark
```

This library provides a comprehensive set of tools for fitting spectral lines in astronomical data. It is designed to 
offer a **user-friendly workflow** that scales seamlessly from **single spectra** to **large datasets**.  

The package includes modules for **masking data**, **line detection**, **profile fitting**, and **storing results** in 
multiple formats.  

If you found LiMe useful, please remember to cite our [A&A publication](https://doi.org/10.1051/0004-6361/202449224) 📝.

## ✨ Key features

- Integrated and profile-dependent measurements of emission and absorption line fluxes.  
- Support for including pixel-level uncertainty spectra in calculations.  
- A flexible configuration system for multi-component profile fitting.  
- Automated and manual tools to verify line detections.  
- Interactive plots for visual inspection of inputs and results.  
- Line labeling conventions compatible with [PyNeb](http://research.iac.es/proyecto/PyNeb/).  
- Flexible output results format, including `.txt`, `.pdf`, multi-page `.fits`, `.asdf`, and `.xlsx` files.
- Support for both long-slit and integrated fiels spectrocopic cubes *.fits* files. 

:::{tip} **Where to find what you need**
- The **Introduction** section provides detailed descriptions of *LiMe*’s main components.  
- The **Guides** section offers shorter tutorials for specific tasks.  
- The **Explanations** section discusses LiMe’s configuration and results from a scientific point of view.  
- The **Reference** section contains comprehensive indexes of *LiMe* functions.  
:::

## ✨ LiMe v2 release:

After a year of improvements, we’re releasing LiMe v2, with a more complete line database and a robust bands generation 
model for a more concise workflow:

```python
import lime

# Data location
data_folder = '../doc_notebooks/0_resources/'

# Load the configuration file
cfgFile = f'{data_folder}/long_slit.toml'
obs_cfg = lime.load_cfg(cfgFile)

# Load the spectrum file
fits_file = f'{data_folder}/spectra/gp121903_osiris.fits'
spec = lime.Spectrum.from_file(fits_file, instrument='osiris', 
                               redshift=obs_cfg['osiris']['gp121903']['z'])

# Generate the object lines table
lines_frame = spec.retrieve.lines_frame(band_vsigma=100, automatic_grouping=True,
                                        fit_cfg=obs_cfg, obj_cfg_prefix='gp121903_osiris')

# Confirm the presence of lines using intensity thresholding
match_lines = spec.infer.peaks_troughs(lines_frame, emission_shape=True)

# Measure the lines
spec.fit.frame(match_lines, obs_cfg, obj_cfg_prefix='gp121903_osiris')

# Measure the lines
spec.fit.frame(match_lines, obs_cfg, obj_cfg_prefix='gp121903_osiris')

# Plot the results
spec.plot.spectrum(log_scale=True, rest_frame=True)
spec.plot.grid()

# Save the results
spec.save_frame('./gp121903_lines_frame.txt')
```

## 📺 Video introduction:

Many commands from the [**LiMe v1 video tutorial**](https://www.youtube.com/embed/k733YS84cUg) are outdated, but it still 
provides a good insight on LiMe workflow.

<iframe
  src="https://www.youtube.com/embed/k733YS84cUg"
  title="YouTube video"
  style="width:100%; height:auto; aspect-ratio:16 / 9; border:0;"
  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
  allowfullscreen>
</iframe>


## 🌌 🔭 SpecSy alpha release:

In our ongoing effort to improve the accessibility of astronomical software and data, we are now releasing the alpha 
version of SpecSy—an open-source, web-based spectroscopic workspace that includes several LiMe utilities.
Try it here: [https://specsy.streamlit.app/](https://specsy.streamlit.app/)

------------------------------------------------------------------------

## 📚 Table of Contents

```{toctree}
---
maxdepth: 1
caption: Introduction
---
1_introduction/0_installation
1_introduction/1_observations
1_introduction/2_line_labels
1_introduction/3_line_bands
1_introduction/4_lines_database
1_introduction/5_fitting_configuration
```


```{toctree} 
---
maxdepth: 2
caption: Guides
---
2_guides/0_creating_observations
2_guides/1_prepare_line_bands
2_guides/2_manual_bands_adjustement
2_guides/3_continuum_fitting
2_guides/4_line_detection
2_guides/5_multifile-line
2_guides/6_ifu_spatial_masking
2_guides/7_ifu_line_fitting
2_guides/8_ifu_results
```

```{toctree} 
---
maxdepth: 2
caption: Explanations
---
3_explanations/0_measurements.rst

```

```{toctree} 
---
maxdepth: 1
caption: Reference
---
4_references/0_API.md
4_references/1_changelog.md

```
