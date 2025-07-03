import timeit

# Simulated config dictionary
fit_conf = {
    'O3_5007A_b': 'O3_5007A+O3_5007A_k-1+He1_5016A',
    'O3_5007A_m': 'O3_5007A+O3_5007A_k-1'
}


# Setup code as a string
setup_code = """
from lime.transitions import Transition, Line
from __main__ import fit_conf
"""

# Statements to time
stmt_old = "Line('O3_5007A', fit_conf)"
stmt_new = "Transition.from_db('O3_5007A', fit_conf)"

# Measure
repeats = 10_000
time_old = timeit.timeit(stmt=stmt_old, setup=setup_code, number=repeats)
time_new = timeit.timeit(stmt=stmt_new, setup=setup_code, number=repeats)

print(f"Line (old) time: {time_old/repeats:.6f} sec ({time_old:0.2f} sec total {repeats} runs)")
print(f"Transition (new) time: {time_new/repeats:.6f} sec ({time_new:0.2f} sec total {repeats} runs)")
