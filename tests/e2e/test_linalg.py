import numpy as np

from pydsl.frontend import compile
from pydsl.memref import DYNAMIC, MemRefFactory
from pydsl.tensor import TensorFactory
from pydsl.type import F64, UInt64
import pydsl.linalg as linalg
from helper import multi_arange, run, log

TensorF64 = TensorFactory((DYNAMIC,), F64)
MemRefF64 = MemRefFactory((DYNAMIC,), F64)
TensorUI64 = TensorFactory((DYNAMIC,), UInt64)
MemRefUI64 = MemRefFactory((DYNAMIC,), UInt64)


def test_linalg_exp():
    @compile(dump_mlir=True, dump_mlir_passes=True)
    def f(t1: TensorF64) -> TensorF64:
        return linalg.exp(t1)

    t1 = np.asarray([i for i in range(10)], np.float64)
    cor_res = np.exp(t1)
    assert np.allclose(f(t1), cor_res)
    # CHECK: SUCCESS: test_linalg_exp
    log("SUCCESS: test_linalg_exp")


def test_linalg_log():
    @compile()
    def f(t1: TensorF64) -> TensorF64:
        return linalg.log(t1)

    t1 = np.asarray([i for i in range(1, 10)], np.float64)
    assert np.allclose(f(t1), np.log(t1))


def test_linalg_abs():
    @compile()
    def f(t1: TensorF64) -> TensorF64:
        return linalg.abs(t1)

    t1 = np.asarray([i for i in range(10)], np.float64)
    assert np.allclose(f(t1), np.abs(t1))


def test_linalg_ceil():
    @compile()
    def f(t1: TensorF64) -> TensorF64:
        return linalg.ceil(t1)

    t1 = np.asarray([i for i in range(10)], np.float64)
    assert np.allclose(f(t1), np.ceil(t1))


def test_linalg_floor():
    @compile()
    def f(t1: TensorF64) -> TensorF64:
        return linalg.floor(t1)

    t1 = np.asarray([i for i in range(10)], np.float64)
    assert np.allclose(f(t1), np.floor(t1))


def test_linalg_negf():
    @compile()
    def f(t1: TensorF64) -> TensorF64:
        return linalg.negf(t1)

    t1 = np.asarray([i for i in range(10)], np.float64)
    assert np.allclose(f(t1), np.negative(t1))


def test_linalg_round():
    @compile()
    def f(t1: TensorF64) -> TensorF64:
        return linalg.round(t1)

    t1 = np.asarray([i for i in range(10)], np.float64)
    assert np.allclose(f(t1), np.round(t1))


def test_linalg_sqrt():
    @compile()
    def f(t1: TensorF64) -> TensorF64:
        return linalg.sqrt(t1)

    t1 = np.asarray([i for i in range(10)], np.float64)
    assert np.allclose(f(t1), np.sqrt(t1))


def test_linalg_rsqrt():
    @compile()
    def f(t1: TensorF64) -> TensorF64:
        return linalg.rsqrt(t1)

    t1 = np.asarray([i for i in range(1, 10)], np.float64)
    assert np.allclose(f(t1), np.reciprocal(np.sqrt(t1)))


def test_linalg_square():
    @compile()
    def f(t1: TensorF64) -> TensorF64:
        return linalg.square(t1)

    t1 = np.asarray([i for i in range(10)], np.float64)
    assert np.allclose(f(t1), np.square(t1))


def test_linalg_tanh():
    @compile()
    def f(t1: TensorF64) -> TensorF64:
        return linalg.tanh(t1)

    t1 = np.asarray([i for i in range(10)], np.float64)
    assert np.allclose(f(t1), np.tanh(t1))


# Numpy doesn't have erf. Scipy is needed.
# def test_linalg_erf():
#     @compile()
#     def f(t1: TensorF64) -> TensorF64:
#         return linalg.erf(t1)

#     t1 = np.asarray([i for i in range(10)], np.float64)
#     assert np.allclose(f(t1), np.erf(t1))

def test_multiple_unary():
    @compile(dump_mlir=True, dump_mlir_passes=True)
    def f(t1: TensorF64) -> TensorF64:
        t2 = linalg.exp(t1)
        t3 = linalg.sqrt(t1)
        return linalg.add(t2, t3)

    n1 = multi_arange((50,), np.float64) / 10
    cor_res = np.exp(n1) + np.sqrt(n1)
    assert np.allclose(f(n1), cor_res)


if __name__ == "__main__":
    run(test_linalg_exp)
    # run(test_linalg_log)
    # run(test_linalg_abs)
    # run(test_linalg_ceil)
    # run(test_linalg_floor)
    # run(test_linalg_negf)
    # run(test_linalg_round)
    # run(test_linalg_sqrt)
    # run(test_linalg_rsqrt)
    # run(test_linalg_square)
    # run(test_linalg_tanh)
    # run(test_multiple_unary)
