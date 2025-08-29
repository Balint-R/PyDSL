from pydsl.memref import *
from pydsl.type import *
from pydsl.frontend import compile
from pydsl.scf import range

MemRef64 = MemRefFactory((1000,), UInt32)

@compile(locals(), dump_mlir=True)
def hello(m: MemRef64) -> MemRef64:
    # for i in range(Index(3), Index(7), Index(2)):
    #     m[Index(0), Index(0)] = m[Index(0), Index(0)] + UInt32(i)
    
    return m

# DO NOT PASS IN THE ARRAY DIRECTLY LIKE THIS!
res = hello(np.zeros((1000,), dtype=np.uint32))  # This could just segfault

print(res) # Garbage array if hello didn't segfault
