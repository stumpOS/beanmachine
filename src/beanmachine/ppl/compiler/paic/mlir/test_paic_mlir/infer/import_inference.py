import ast
import inspect
import typing

# this decorator must identify two methods in the class:
# 'create_world'
# infer: [RVIdentifier] -> Tensor
# The effect of this decorator is that the infer will be modified to instead call the lowered function.
# The create_world method will not be changed from the user's perspective but this decorator will update that
# create world method
import beanmachine.ppl.compiler.bm_to_bmg
import beanmachine.ppl.compiler.paic.mlir.test_paic_mlir.to_paic_ast
import beanmachine.ppl.compiler.paic.mlir.test_paic_mlir.utils


def import_inference(callable: typing.Callable) -> typing.ClassVar:
    """
    Given:
    (1) a function that
        (a) accepts queries, observations. The only read read operation is when creating the world
        (b) creates a world from queries and observations
    (2) arguments for function (1)

    This function will do the following:
    (1) Generate (note that this will occur in the C++ layer):
        (a) [FROM FUNC] a world interface IWorld optimized for the given function
        (b) [FROM FUNC] a lowered function that accepts an IWorld
        (c) [FROM ARGS] an implementation of the world identified in (1b)
    (2) invoke the generated function with an instance of the generated implementation
    """
    def wrapper(*args, **kwargs):
        lines, _ = inspect.getsourcelines(callable)
        source = "".join(beanmachine.ppl.compiler.paic.mlir.test_paic_mlir.utils._unindent(lines))
        module = ast.parse(source)
        funcdef = module.body[0]
        # TODO: collect ASTs of query methods
        to_paic = beanmachine.ppl.compiler.paic.mlir.test_paic_mlir.to_paic_ast.paic_ast_generator()
        globals = beanmachine.ppl.compiler.bm_to_bmg._get_globals(callable)
        python_function = to_paic.python_ast_to_paic_ast(funcdef, globals)
        # TODO: pass the paic ast to the import function instead
        arg = float(args[0])
        result = mb.to_metal(python_function, arg)
        return result
    return callable
