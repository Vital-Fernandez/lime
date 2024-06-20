import lime

# Check version >= 1.0.6
print(lime.__version__)

# File location
megara_cube_address = '/home/vital/Astrodata/Karla_MEGARA/NGC5471_datacube_LR-V_3600_scale03_drp_nosky.fits'

# Create LiMe cube
ngc5471 = lime.Cube.from_file(megara_cube_address, instrument='megara', redshift=0.00091)
ngc5471.unit_conversion(units_flux='FLAM', norm_flux=1e-18)

# Interactive plot (right click change spaxel)
ngc5471.check.cube('He1_5876A')

# Generate a spatial mask
mask_file = './He1_5876A_3level_mask.fits'
ngc5471.spatial_masking('He1_5876A', param='SN_line', contour_pctls=[56, 80, 96], output_address=mask_file)
ngc5471.plot.cube('He1_5876A', masks_file=mask_file)

# Adjust mask manually (middle button add or remove spaxels from mask)
ngc5471.check.cube('He1_5876A', masks_file=mask_file)

