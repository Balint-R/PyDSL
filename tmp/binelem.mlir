module {
    func.func @f(%arg0: tensor<32x64xi32>, %arg1: memref<32x64xf64>, %arg2: memref<32x64xf32>) {
        linalg.elemwise_binary {cast = #linalg.type_fn<cast_signed>, fun = #linalg.binary_fn<add>}
            ins(%arg0, %arg1 : tensor<32x64xi32>, memref<32x64xf64>)
            outs(%arg2 : memref<32x64xf32>)
        return
    }
}
