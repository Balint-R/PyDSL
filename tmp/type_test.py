from pydsl.type import F32

class A:
    class B:
        pass

x = A.B()
print(type(x))
print(type(x).__qualname__)
print(type(x).__name__)

print(F32)
print(F32.__qualname__)
print(F32.__name__)
