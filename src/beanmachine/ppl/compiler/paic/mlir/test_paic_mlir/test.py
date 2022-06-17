import math

from beanmachine.ppl.compiler.paic.mlir.test_paic_mlir.to_metal import to_metal

x = math.pow

@to_metal
def foo(p1:float) -> float:
    i0 = p1*p1
    i1 = math.pow(i0, 2.0)
    return i1


if __name__ == "__main__":
    call = foo(4)
    print("mlir returned " + str(call))
