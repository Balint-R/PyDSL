from pydsl.frontend import compile
from pydsl.type import UInt32

@compile(dump_mlir=True)
def f():
    a = 2
    b = UInt32(3)

f()
