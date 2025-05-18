import lime

fname = '/home/vital/PycharmProjects/lime/tests/baseline/manga_lines_log.txt'
df = lime.load_frame(fname)
bands = lime.bands_from_measurements(fname)

# print(df)
print(bands)
for line in ['O2_3726A_b', 'H1_3889A_m', 'O3_4363A', 'H1_4861A_b', 'H1_6563A_b']:
    line = lime.Line(line, band=bands)
    print(line.label, line.blended_check, line.merged_check, line.group_label, line.transition_comp)