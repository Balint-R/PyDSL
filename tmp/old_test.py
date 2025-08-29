def g(*args):
    print("args =", args)


cls = type("Foo", (), {"f": g})

cls.f(123, 456)
cls().f(123, 456)


class Bar:
    f = g


Bar.f(123, 456)
Bar().f(123, 456)


class Baz:
    @classmethod
    def f(*args):
        print("args =", args)


Baz.f(123, 456)
Baz().f(123, 456)
