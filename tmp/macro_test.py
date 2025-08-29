from pydsl.type import UInt32

print(UInt32.on_Call)
print(UInt32(5).on_Call)

class Foo:
    def f(*args):
        print(*args)

    @staticmethod
    def g(*args):
        print(*args)

x = Foo()

# print(x.f)
# print(x.f(123))
# print(x.f)

# print(Foo.f)
# print(Foo().f)
# print(Foo.g)
# print(Foo().g)

# print(isinstance(int, int))
