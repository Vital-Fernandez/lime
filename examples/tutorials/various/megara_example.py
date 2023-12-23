import lime

# Check version >= 0.9.96
print(lime.__version__)

# File location
megara_cube_address = '../../sample_data/spectra/NGC5471_datacube_LR-R_900_scale03_drp_nosky.fits'

# Create LiMe cube
ngc5471 = lime.Cube.from_file(megara_cube_address, instrument='megara', redshift=0.00091, norm_flux=1)

# Interactive plot (right click change spaxel)
ngc5471.check.cube('H1_6563A')

# Generate a spatial mask
mask_file = './Halpha_3level_mask.fits'
ngc5471.spatial_masking('H1_6563A', param='SN_line', contour_pctls=[56, 80, 96], output_address=mask_file)
ngc5471.plot.cube('H1_6563A', masks_file=mask_file)

# Adjust mask manually (middle button add or remove spaxels from mask)
ngc5471.check.cube('H1_6563A', masks_file=mask_file)

