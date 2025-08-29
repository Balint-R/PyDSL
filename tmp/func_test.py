
@compile()
def f():
    if f:
        a = 2
    else:
        a = 1
        b = 4

    print(a)
    print(b)

@compile()
def f():
    for i in range(5):
        a = i

    print(a)

    for i in range(0):
        b = i

    print(b)

    # for i in range(n):
    #     for j in range(i):

    for i in range(n):
        c = i

    if c is None:
        pass
    else:
        print(c)
