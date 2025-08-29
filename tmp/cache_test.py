from functools import cache

@cache
def f(a: int, b: int = 0):
    print(a, b)
    return a, b

f(12)
f(12, 0)
