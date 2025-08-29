import inspect
from pydsl.macro import CallMacro, Evaluated, Compiled
from pydsl.protocols import ToMLIRBase, SubtreeOut, OpView
from pydsl.frontend import compile

class LoopMacro(CallMacro):
    @staticmethod
    def signature() -> inspect.Signature:
        def f(visitor: ToMLIRBase, target: Compiled, index: Evaluated): ...
        return inspect.signature(f)

    @staticmethod
    def __call__(visitor: ToMLIRBase, target: Compiled, index: Evaluated) -> OpView:
        if hasattr(target, "loops"):
            return target.loops[index]
        return target.operation.results[index]

get_loop = LoopMacro()

# @compile(dump_mlir=True)
# def f():
