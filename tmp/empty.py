from pydsl.frontend import compile
import numpy as np

from pydsl.type import Index
from pydsl.memref import DYNAMIC
from pydsl.tensor import Tensor, emptyT, TensorFactory
from pydsl.type import F32

TensorF32 = TensorFactory((4, 8, DYNAMIC), F32)
TensorF32_1 = Tensor[F32, 4, 8, DYNAMIC]


@compile(dump_mlir=True)
def f(t1: Tensor[F32, DYNAMIC]) -> Tensor[F32, DYNAMIC]:
    a = F32(4)
    t1.cast((12, 34))
    t1 = emptyT((5,), F32)
    return t1


n1 = np.arange(10, dtype=np.float32)
res = f(n1)
print(res)
