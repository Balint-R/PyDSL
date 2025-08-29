from mlir.ir import Context, Location, Module

code = r"""module {
  func.func public @f1() -> i16 {
    %0:2 = call @f2() : () -> (i16, i16)
    %1 = arith.addi %0#0, %0#1 : i16
    return %1 : i16
  }
  func.func public @f2() -> (i16, i16) {
    %c12_i16 = arith.constant 12 : i16
    %c34_i16 = arith.constant 34 : i16
    return %c12_i16, %c34_i16 : i16, i16
  }
}
"""

print(code)

with Context() as ctx, Location.unknown():
    module = Module.parse(code)
    print(module)
