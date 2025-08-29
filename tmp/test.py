from pydsl.frontend import compile
from pydsl.scf import range as srange
from pydsl.type import F32, F64

@compile(dump_mlir=True)
def f(a: F32, b: F64) -> F64:
    return b + a

print(f(1, 2))
