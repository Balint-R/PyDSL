module {
    func.func @f(%arg0: memref<16x32x64xf32>, %arg1: memref<?xf64>) {
        linalg.reduce
            ins(%arg0: memref<16x32x64xf32>)
            outs(%arg1: memref<?xf64>)
            dimensions = [1, 2]
            (%in: f32, %out: f64) {
                %0 = arith.extf %in : f32 to f64
                %1 = arith.addf %out, %0: f64
                linalg.yield %1: f64
            }
      return
    }
}
