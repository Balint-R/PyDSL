import ast

code = r"""@compile()
def calc[T, N, M](mat: Tensor[T, N, M]) -> Tensor[T, M, N]: ...
calc[F32, 1000, 500](mat1)
calc[F64, DYNAMIC, 4000](mat2)
"""

print(ast.dump(ast.parse(code), indent=4))
