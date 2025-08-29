"builtin.module"() ({
  "func.func"() <{function_type = (memref<16xf32>, memref<16x64xf64>) -> (), sym_name = "f"}> ({
  ^bb0(%arg0: memref<16xf32>, %arg1: memref<16x64xf64>):
    "linalg.broadcast"(%arg0, %arg1) <{dimensions = array<i64: 1>}> ({
    ^bb0(%arg2: f32, %arg3: f64):
      "linalg.yield"(%arg3) : (f64) -> ()
    }) : (memref<16xf32>, memref<16x64xf64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
