import ast
import inspect

code = r"""
@abc
@xyz
def f(a, b):
    return a + b
"""

def f(a, b):
    return a + b

print(inspect.getsource(f))

tree1 = ast.parse(code)
tree1 = tree1.body[0]
print(type(tree1))
print(ast.dump(tree1, indent=4))
