import numpy as np
import pandas as pd
from pathlib import Path
import lime
from lime.transitions import latex_from_label, label_components, Line

parent_mask_address = Path(lime._dir_path)/f'resources/parent_mask.txt'

parend_band = lime.load_log(parent_mask_address)

# for line in parend_band.index:
#     band_line = parend_band.loc[line, 'w1':'w6'].values
#     check = np.all(np.diff(band_line) >= 0)
#
#     if check == False:
#         print(line, band_line)

# lime.save_line_log(parend_band, parent_mask_address)

# ion, wave, units_wave, kinem = label_components(parend_band.index.values)
# latex = latex_from_label(None, ion, wave, units_wave, kinem)
#
# label = 'O3_5007A'
# fit_conf = {'O3_5007A_b': 'O3_5007A-O3_5007A_k1',
#             'O3_5007A_m': 'O3_5007A-He1_5016A'}
#
# o3_b = Line('O3_5007A_b', fit_conf=fit_conf)
# o3 = Line('O3_5007A', fit_conf=fit_conf)
# o3_m = Line('O3_5007A_m', fit_conf=fit_conf)
#
# print(o3.latex, len(o3.latex))
# print(o3_b.latex, len(o3_b.latex))
# print(o3_m.latex, len(o3_m.latex))

d = {'col1': [0, 1, np.nan, None], 'col2': pd.Series([2, 3], index=[2, 3])}
df = pd.DataFrame(data=d, index=[0, 1, 2, 3])

print(df)

lime.save_log(df, 'test.txt')
df_load = lime.load_log('test.txt')
print('\n', df_load)

