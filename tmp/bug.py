from pydsl.frontend import compile
from pydsl.type import UInt16, UInt32

@compile()
class Module:
    def my_add(a: UInt16, b: UInt16) -> UInt32:
        return a + b
    
    def triple_add(a: UInt16, b: UInt16, c: UInt16) -> UInt32:
        return my_add(a, b) + c
    
assert Module.triple_add(20000, 30000, 40000) == 20000 + 30000 + 40000
