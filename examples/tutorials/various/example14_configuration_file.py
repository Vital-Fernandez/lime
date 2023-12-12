import pyneb as pn
from matplotlib import pyplot as plt

S2 = pn.Atom('S', 2)

f, ax = plt.subplots(figsize=(8, 6))
S2.plotGrotrian(tem=1e4, den=1e2, unit = 'eV', ax=ax)
plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib
# # Sample data
# x = np.linspace(0, 10, 100)
# y = np.sin(x)
# from matplotlib import rcParams
#
# path_style = r'D:\Pycharm Projects\lime\src\lime\resources\styles\lime_default.mplstyle'
#
# with plt.style.context(path_style):
#     with plt.rc_context(None):
#
#         fig, ax = plt.subplots()
#
#         prop_cycle = ax._get_lines.prop_cycler
#
#         # Extract the colors from the cycler
#         rcParams['axes.prop_cycle'].by_key()
#         colors = rcParams['axes.prop_cycle'].by_key()['color']
#
#         rcParams['axes.prop_cycle'].by_key()
#
#         for i, color in enumerate(colors):
#             print(i, colors[i])
#             x_range = np.arange(10)
#             y_range = np.full(x_range.shape, i)
#             ax.plot(x_range, y_range, color=color)
#
#         ax.set_title('Sin Wave with Custom Style and High DPI')
#         ax.set_xlabel('X Axis')
#         ax.set_ylabel('Y Axis')
#         plt.show()


# with plt.style.context(['grayscale', './presentation.mplstyle']):
#     with plt.rc_context({'axes.titlecolor': 'red', 'figure.dpi': 150}):
#         fig, ax = plt.subplots()
#         ax.plot(x, y)
#         ax.set_title('Sin Wave with Custom Style and High DPI')
#         ax.set_xlabel('X Axis')
#         ax.set_ylabel('Y Axis')
#         plt.show()
#
# # for i, style in enumerate(list_styles):
#     print(i, style)
#
#     with plt.rc_context({}):
#         with plt.style.context(style):
#             fig, ax = plt.subplots()
#             ax.plot(x, y)
#             ax.set_title(style)
#             ax.set_xlabel('X Axis')
#             ax.set_ylabel('Y Axis')
#             plt.show()



# # Using rc_context to temporarily set the DPI
# with plt.rc_context({'figure.dpi': 150}):
#     fig, ax = plt.subplots()
#     ax.plot(x, y)
#     ax.set_title('Sin Wave with Custom Style and High DPI')
#     ax.set_xlabel('X Axis')
#     ax.set_ylabel('Y Axis')
#     plt.show()
#
# # Using rc_context to temporarily set the DPI
# with plt.style.context('grayscale'):
#     with plt.rc_context({'axes.titlecolor': 'red', 'figure.dpi': 150}):
#         fig, ax = plt.subplots()
#         ax.plot(x, y)
#         ax.set_title('Sin Wave with Custom Style and High DPI')
#         ax.set_xlabel('X Axis')
#         ax.set_ylabel('Y Axis')
#         plt.show()
#
#
# # Using rc_context to temporarily set the DPI
# with plt.rc_context({'axes.titlecolor': 'red', 'figure.dpi': 150}):
#     with plt.style.context('grayscale'):
#         fig, ax = plt.subplots()
#         ax.plot(x, y)
#         ax.set_title('Sin Wave with Custom Style and High DPI')
#         ax.set_xlabel('X Axis')
#         ax.set_ylabel('Y Axis')
#         plt.show()
