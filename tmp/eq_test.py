class Foo:
    a: int

    def __init__(self, a: int):
        self.a = a

x = Foo(4)
y = Foo(4)
print(x == y)
s = {x}


from enum import Enum

import mlir.dialects.gpu as gpu

class GPU_AddrSpace(Enum):
    Global = gpu.AddressSpace.Global
    Workgroup = gpu.AddressSpace.Workgroup
    Private = gpu.AddressSpace.Private

    def lower(self):
        print(type(self))
        print(type(self.value))
        return str(self.value)

a = GPU_AddrSpace.Global
print(a.lower())
