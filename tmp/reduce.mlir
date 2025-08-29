module {
    func.func @f(%arg0: memref<16x32x64xf32>, %arg1: memref<16x64xf32>) {
        linalg.reduce
            ins(%arg0: memref<16x32x64xf32>)
            outs(%arg1: memref<16x64xf32>)
            dimensions = [1]
            (%in: f32, %out: f32) {
                %0 = arith.addf %out, %in: f32
                linalg.yield %0: f32
            }
      return
    }
}
