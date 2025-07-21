import collections.abc as cabc
import ast
import typing
from functools import cache
from pydsl.macro import CallMacro, Compiled, Evaluated, MethodType

import mlir.dialects.tensor as mlir_tensor
import mlir.ir as mlir
from mlir.ir import DenseI64ArrayAttr, OpView, RankedTensorType, Value

from pydsl.memref import (
    UsesRMRD,
    RuntimeMemrefShape,
    slices_to_mlir_format,
    subtree_to_slices,
)

from pydsl.type import (
    Index,
    Lowerable,
    lower,
    lower_flatten,
    lower_single,
    SupportsIndex,
    Tuple,
)
from pydsl.protocols import SubtreeOut, ToMLIRBase

DYNAMIC = -9223372036854775808

# used for the virtual static-typing in PyDSL
Dynamic = typing.Literal[-9223372036854775808]

# based on example in PEP 646: https://peps.python.org/pep-0646/
# TODO: these currently are unused
DType = typing.TypeVar("DType")
Shape = typing.TypeVarTuple("Shape")


class Tensor(typing.Generic[DType, *Shape], UsesRMRD):
    """
    TODO: this Tensor type is limited to ranked versions for now.
    """

    value: Value
    shape: tuple[int] = None
    element_type: Lowerable = None
    offset: int = None
    strides: tuple[int] = None
    _default_subclass_name = "AnnonymousTensorSubclass"
    _supported_mlir_type = [
        mlir.IntegerType,
        mlir.F16Type,
        mlir.F32Type,
        mlir.F64Type,
        mlir.IndexType,
        mlir.ComplexType,
    ]

    @staticmethod
    @cache
    def class_factory(
        shape: tuple[int], element_type, name=_default_subclass_name
    ):
        """
        Create a new subclass of Tensor dynamically with the specified
        dimensions and type
        """
        # TODO: this check cannot be done right now because types can't be
        # lowered outside of MLIR context
        # if len(lower(element_type)) != 1: raise TypeError(f"Tensor cannot
        # store composite types, got {element_type} which lowers to
        # {lower(element_type)}")

        if not isinstance(shape, cabc.Iterable):
            raise TypeError(
                f"Tensor requires shape to be iterable, got {type(shape)}"
            )

        return type(
            name,
            (Tensor,),
            {
                "shape": tuple(shape),
                "element_type": element_type,
                # tensor seems to get lowererd to memref<..., strided<[?, ?, ?], offset: ?>>
                "offset": DYNAMIC,
                "strides": tuple([DYNAMIC] * len(shape)),
            },
        )

    # Convenient alias
    get = class_factory

    def __init__(self, rep: OpView | Value) -> None:
        mlir_element_type = lower_single(self.element_type)
        if not any([
            type(mlir_element_type) is t for t in self._supported_mlir_type
        ]):
            raise NotImplementedError(
                f"having a Tensor with DType {self.element_type.__qualname__} "
                f"is not supported"
            )

        if isinstance(rep, OpView):
            rep = rep.result

        if (rep_type := type(rep.type)) is not RankedTensorType:
            raise TypeError(f"{rep_type} cannot be casted as a Tensor")

        if not all([
            self.shape == tuple(rep.type.shape),
            lower_single(self.element_type) == rep.type.element_type,
        ]):
            raise TypeError(
                f"expected shape {'x'.join([str(sh) for sh in self.shape])}"
                f"x{lower_single(self.element_type)}, got OpView with shape "
                f"{'x'.join([str(sh) for sh in rep.type.shape])}"
                f"x{rep.type.element_type}"
            )

        self.value = rep

    def lower(self) -> tuple[Value]:
        return (self.value,)

    @classmethod
    def lower_class(cls) -> tuple[mlir.Type]:
        if not all([cls.shape, cls.element_type]):
            e = TypeError(
                "attempted to lower Tensor without defined dims or type"
            )
            if (clsname := cls.__name__) != Tensor._default_subclass_name:
                e.add_note(f"hint: class name is {clsname}")
            raise e

        return (
            RankedTensorType.get(
                list(cls.shape), lower_single(cls.element_type)
            ),
        )

    # TODO: potential dead code. MLIR already compute all the dims for us if we
    # pass the input tensor as the output tensor as well. I feel this can still
    # be useful if the end PyDSL user wants to get the shape though.
    # Update: on_getitem and on_setitem use this function now.
    @property
    def runtime_shape(self) -> RuntimeMemrefShape:
        """
        Return the shape of the tensor as it exists at runtime.

        If one of the dimension sizes is dynamic, a tensor.dim operator is
        returned instead for that dimension.
        """
        return [
            (
                d
                if d != DYNAMIC
                else mlir_tensor.DimOp(
                    lower_single(self), lower_single(Index(i))
                )
            )
            for i, d in enumerate(self.shape)
        ]

    def on_getitem(
        self: typing.Self, visitor: "ToMLIRBase", slice: ast.AST
    ) -> SubtreeOut:
        key_list = subtree_to_slices(visitor, visitor.visit(slice))
        dim = len(self.shape)

        # If all indices are integers, not slices, do an extract op
        if len(key_list) == dim and all(
            isinstance(key, SupportsIndex) for key in key_list
        ):
            key_list = lower_flatten([Index(key) for key in key_list])
            rep = mlir_tensor.extract(lower_single(self), key_list)
            return self.element_type(rep)

        # Otherwise, do an extract_slice op
        lo_list, size_list, step_list = slices_to_mlir_format(
            key_list, self.runtime_shape
        )

        # We make the result a tensor with all dynamic dimensions
        result_type = TensorFactory(tuple([DYNAMIC] * dim), self.element_type)
        dynamic_i64_attr = DenseI64ArrayAttr.get([DYNAMIC] * dim)

        rep = mlir_tensor.extract_slice(
            result_type.lower_class()[0],
            lower_single(self),
            lo_list,
            size_list,
            step_list,
            dynamic_i64_attr,
            dynamic_i64_attr,
            dynamic_i64_attr,
        )
        return result_type(rep)

    def on_setitem(
        self: typing.Self,
        visitor: "ToMLIRBase",
        slice: ast.AST,
        value: ast.AST,
    ) -> SubtreeOut:
        value_st = visitor.visit(value)
        key_list = subtree_to_slices(visitor, visitor.visit(slice))
        dst_dim = len(self.shape)

        # If all indices are integers, not slices, do an insert op
        if len(key_list) == dst_dim and all(
            isinstance(key, SupportsIndex) for key in key_list
        ):
            value_mlir = lower_single(self.element_type(value_st))
            key_list = lower_flatten([Index(key) for key in key_list])
            rep = mlir_tensor.insert(value_mlir, lower_single(self), key_list)
            self.value = rep
            return rep

        # Otherwise, do an insert_slice op
        lo_list, size_list, step_list = slices_to_mlir_format(
            key_list, self.runtime_shape
        )
        src_dim = len(value_st.shape)

        if dst_dim != src_dim:
            raise TypeError(
                "trying to insert_slice with tensors of different ranks"
            )

        # We make all offsets and strides dynamic
        dynamic_i64_attr = DenseI64ArrayAttr.get([DYNAMIC] * src_dim)

        # We use static dimensions for the shape of the source tensor whenever possible.
        # This is necessary to not cause an error (can't use tensor of static shape
        # if the op expects dynamic shape).
        src_size_i64_attr = DenseI64ArrayAttr.get(value_st.shape)
        size_list = [
            size_list[i]
            for i in range(src_dim)
            if value_st.shape[i] == DYNAMIC
        ]

        rep = mlir_tensor.insert_slice(
            lower_single(value_st),
            lower_single(self),
            lo_list,
            size_list,
            step_list,
            dynamic_i64_attr,
            src_size_i64_attr,
            dynamic_i64_attr,
        )
        self.value = rep
        return rep

    def __class_getitem__(cls, args: tuple):
        if not isinstance(args, tuple):
            args = (args,)
        
        if len(args) < 2:
            raise TypeError(
                f"Tensor expected at least 2 Generic arguments, got {args}"
            )
        
        dtype = args[0]
        shape = args[1:]

        return cls.class_factory(tuple(shape), dtype)

    @classmethod
    def on_class_getitem(
        cls, visitor: ToMLIRBase, slice: ast.AST
    ) -> SubtreeOut:
        # TODO: directly copied from Memref. Make a helper function somwhere
        # to avoid duplicating code.
        match slice:
            case ast.Tuple(elts=elts):
                args = [visitor.resolve_type_annotation(e) for e in elts]
            case t:
                args = [visitor.resolve_type_annotation(t)]

        if len(args) < 2:
            raise TypeError(
                f"Tensor expected at least 2 Generic arguments, got {args}"
            )

        dtype = args[0]
        shape = args[1:]

        return cls.class_factory(tuple(shape), dtype)
    
    @CallMacro.generate(method_type=MethodType.INSTANCE)
    def cast(visitor: "ToMLIRBase", self: typing.Self, shape: Evaluated):
        """
        Convert a tensor from one type to an equivalent type without changing
        any data elements. The resulting tensor type will have the same element
        type. shape is the shape of the new tensor and must be known at compile
        time. For any constant dimensions of shape, the input tensor must
        actually have that dimension at runtime, otherwise the operation is
        invalid.

        Note: this function only returns a tensor with the updated type, it
        does not modify the type of the input tensor.

        Example:

        def f(t1: Tensor[F32, DYNAMIC, 32, 5]) -> Tensor[F32, 64, 32, DYNAMIC]:
            # Only valid if the first dimension of t1 is always 64
            t2 = t1.cast((64, 32, DYNAMIC))
            return t2
        
        """
        if not isinstance(shape, cabc.Iterable):
            raise TypeError(f"{repr(shape)} is not iterable")
        
        if not all([isinstance(x, int) for x in shape]):
            raise TypeError(
                f"shape should be a tuple of integers known at compile time ",
                f"got {repr(shape)}"
            )
        
        shape = tuple(shape)

        if len(shape) != len(self.shape):
            raise ValueError(
                f"trying to cast a tensor of rank {len(self.shape)} to rank ",
                f"{len(shape)}, ranks should be equal when casting"
            )
        
        for x, y in zip(self.shape, shape):
            if x != y and x != DYNAMIC and y != DYNAMIC:
                raise ValueError(f"incompatible dimensions: {x} and {y}")

        result_type = TensorFactory(shape, self.element_type)
        rep = mlir_tensor.cast(lower_single(result_type), lower_single(self))
        return result_type(rep)


# Convenient alias
TensorFactory = Tensor.class_factory


def verify_tensor_type(t_type: type[Tensor]):
    if not issubclass(t_type, Tensor):
        raise TypeError(
            f"the type being allocated must be a subclass of Tensor, got "
            f"{t_type}"
        )


def verify_dynamics_val(t_type: type[Tensor], dynamics_val: Tuple) -> None:
    dynamics_val = lower(dynamics_val)

    if not isinstance(dynamics_val, cabc.Iterable):
        raise TypeError(f"{repr(dynamics_val)} is not iterable")

    if (actual_dyn := len(dynamics_val)) != (
        target_dyn := t_type.shape.count(DYNAMIC)
    ):
        raise ValueError(
            f"Tensor has {target_dyn} dynamic dimensions to be filled, "
            f"but emptyOp received {actual_dyn}"
        )


# @CallMacro.generate()
def empty(
    visitor: ToMLIRBase, shape: Compiled, dtype: Evaluated
) -> SubtreeOut:
    if not isinstance(shape, Tuple):
        raise TypeError(f"shape should be a Tuple, got {type(shape).__qualname__}")
    
    shape = [Index(i) for i in shape.as_iterable(visitor)]
    rank = len(shape)
    t_type = TensorFactory(tuple([DYNAMIC]*rank), dtype)
    return t_type(mlir_tensor.empty(lower_single(t_type), lower_flatten(shape)))


# Python is a bad language
import pydsl.linalg as linalg

@CallMacro.generate()
def zeros(
    visitor: ToMLIRBase, shape: Compiled, dtype: Evaluated
) -> SubtreeOut:
    if not isinstance(shape, Tuple):
        raise TypeError(f"shape should be a Tuple, got {type(shape).__qualname__}")
    
    print(type(empty))
    emp = empty(visitor, shape, dtype)
    return linalg.fill(visitor, dtype(0), emp)

# @CallMacro.generate()
# def empty(
#     visitor: ToMLIRBase, t_type: Evaluated, dynamics_val: Compiled
# ) -> SubtreeOut:
#     verify_tensor_type(t_type)
#     verify_dynamics_val(t_type, dynamics_val)
#     dynamics_val = [lower_single(Index(i)) for i in lower(dynamics_val)]
#     idx = 0
#     orig_shape = t_type.shape
#     shape = [0] * len(orig_shape)
#     for i in range(len(orig_shape)):
#         if orig_shape[i] == DYNAMIC:
#             shape[i] = dynamics_val[idx]
#             idx += 1
#         else:
#             shape[i] = orig_shape[i]
#     return t_type(tensor.empty(lower_single(t_type), dynamics_val))
