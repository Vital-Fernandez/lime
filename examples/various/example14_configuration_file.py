import toml

def my_func(a, b=2, c=3):

    return a + b + c

with open('conf.toml', 'r') as file:
    cfg = toml.load(file)


print(cfg)
print(cfg["default_line_fitting"])
print('line_detection' in cfg["default_line_fitting"])
print(cfg['servers'])


print(my_func(1))

myDict = {'b': 2, 'c':3}
print(my_func(1, **myDict))

myDict = {'a': 2, 'b': 2, 'c':3}
print(my_func(1, **myDict))
