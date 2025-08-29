module {
  func.func public @f1() -> i16 {
    %0, %1 = call @f2() : () -> (i16, i16)
    %2 = arith.addi %0, %1 : i16
    return %2 : i16
  }
  func.func public @f2() -> (i16, i16) {
    %c12_i16 = arith.constant 12 : i16
    %c34_i16 = arith.constant 34 : i16
    return %c12_i16, %c34_i16 : i16, i16
  }
}
