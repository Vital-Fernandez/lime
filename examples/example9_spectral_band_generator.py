import numpy as np
import pandas as pd
from pathlib import Path
import lime

parent_mask_address = Path(lime._dir_path)/f'resources/parent_mask.txt'

parend_band = lime.load_log(parent_mask_address)

for line in parend_band.index:
    band_line = parend_band.loc[line, 'w1':'w6'].values
    check = np.all(np.diff(band_line) >= 0)

    if check == False:
        print(line, band_line)

# lime.save_line_log(parend_band, parent_mask_address)

