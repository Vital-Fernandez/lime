# LiMe: A LIne MEasuring library

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

This library provides a set of tools to fit lines in astronomical spectra. Its design aims for a user-friendly workflow 
for both single lines and large data sets. The library provides tools for masking, detecting, profile fitting and storing
the results. The output measurements are focused on the gas chemical and kinematic analysis. Check the scientific 
publication at [the arXiv](https://arxiv.org/abs/2405.15072) and the video introduction:

<iframe width="560" height="315" src="https://www.youtube.com/embed/k733YS84cUg" frameborder="0" allow="accelerometer;
autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

Some of the library features are:

- Integrated and profile-dependent emission and absorption line flux measurements.
- The user can include the pixel error spectrum in the calculation.
- The Multi-Component profile parameters can be constrained by the user during the fitting.
- Tools to confirm the presence of lines.
- Interactive plots for the visual appraisal of inputs and outputs.
- Line labels adhere to the [PyNeb](http://research.iac.es/proyecto/PyNeb/) format.
- The measurements can be saved in several file types, including `.txt`, `.pdf`, multi-page `.fits`, `.asdf` and `.xlsx` files.

In order to fine the features you want:

:::{tip}
In order to fine the features you want:
[Go to Installation](1_introduction/0_installation.md#for-developers)
:::

```{admonition} Extra features
:class: dropdown
And this will be hidden!
```

------------------------------------------------------------------------

## 📚 Table of Contents

```{toctree}
---
maxdepth: 1
caption: Introduction
---
1_introduction/0_installation
1_introduction/1_observations
```


```{toctree} 
---
maxdepth: 2
caption: Guides
---
2_guides/0_open_fits_files
2_guides/2_measure_multiple_lines


```

```{toctree} 
---
maxdepth: 2
caption: Explanations
---
3_explanations/0_measurements_description

```

```{toctree} 
---
maxdepth: 1
caption: Reference
---
4_reference/0_changelog

```