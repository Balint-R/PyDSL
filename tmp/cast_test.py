from pydsl.frontend import compile
from pydsl.type import Index, UInt32

@compile()
def f() -> Index:
    a = UInt32(123)
    b = Index(a)
    return b

print(f())
