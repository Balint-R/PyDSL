module {
  func.func public @f(%arg0: memref<?x6xi32>, %arg1: memref<?x?xi32>, %arg2: memref<7x?xi32>){
    linalg.add ins(%arg0, %arg1 : memref<?x6xi32>, memref<?x?xi32>) outs(%arg2 : memref<7x?xi32>)
    return
  }
}
