module {
  func.func public @f(%t1: tensor<?xf64>) -> tensor<?xf64> {
    %0 = linalg.exp ins(%t1 : tensor<?xf64>) outs(%t1 : tensor<?xf64>) -> tensor<?xf64>
    %1 = linalg.sqrt ins(%t1 : tensor<?xf64>) outs(%t1 : tensor<?xf64>) -> tensor<?xf64>
    %2 = linalg.elemwise_binary {cast = #linalg.type_fn<cast_signed>, fun = #linalg.binary_fn<add>} ins(%0, %1 : tensor<?xf64>, tensor<?xf64>) outs(%0 : tensor<?xf64>) -> tensor<?xf64>
    return %2 : tensor<?xf64>
  }
}
