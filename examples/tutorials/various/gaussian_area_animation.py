import numpy as np

# Parameters
amp = 5.0
sigma = 2.0
amp_err = 0.1
sigma_err = 0.05

# Relative error (independent of C)
relative_error = np.sqrt((amp_err / amp)**2 + (sigma_err / sigma)**2)

# Range of C values
C_values = np.linspace(0.1, 10, 5)  # few representative values

# Compute f and f_err for each C
for C in C_values:
    f = C * amp * sigma
    f_err = f * relative_error
    rel_err_computed = f_err / f
    print(f"C = {C:.2f}, f = {f:.2f}, f_err = {f_err:.4f}, relative error = {rel_err_computed:.6f}")
