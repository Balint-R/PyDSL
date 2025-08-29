import numpy as np

from pydsl.frontend import compile
from pydsl.memref import MemRef
from pydsl.scf import range
from pydsl.type import SInt16, SInt32

@compile()
def f(arr: MemRef[SInt32, 8, 4], x: SInt32, y: SInt16) -> SInt32:
    z = x + y
    for i in range(8):
        for j in range(4):
            arr[i, j] += z
    
    return z

arr = np.zeros((8, 4), dtype=np.int32)
z = f(arr, 7, 9)
print(arr)
print(z)
