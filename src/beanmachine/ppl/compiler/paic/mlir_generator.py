import _ast


class MLIRCompileError(Exception):
    def __init__(self, message: str):
        self.__cause__ = message

# class mlir_generator:
#     def __init__(self):
#         self.type_map = {'float': llvmlite.ir.FloatType(), 'double': llvmlite.ir.DoubleType}
#
#     def llvm_function(self, function_def: _ast.FunctionDef, module: llvm.ir.Module) -> llvmlite.ir.Function:
#         ir_param_types = []
#         substitutor: typing.Dict[str, llvmlite.ir.Argument] = {}
#         # validate function
#         if isinstance(function_def, _ast.FunctionDef):
#             for a in function_def.args.args:
#                 if isinstance(a, _ast.arg):
#                     if isinstance(a.annotation, _ast.Name):
#                         if not self.type_map.__contains__(a.annotation.id):
#                             raise MLIRCompileError("all arguments must have types in order to be translated into llvm ir")
#                         ir_param_types.append(self.type_map[a.annotation.id])
#
#             if isinstance(function_def.returns, _ast.Name):
#                 return_type = function_def.returns.id
#                 if not self.type_map.__contains__(return_type):
#                     raise MLIRCompileError("return types must have types in order to be translated into llvm ir")
#                 ir_return_type = self.type_map[function_def.returns.id]
#
#         to_process = []
#         function_signature = llvmlite.ir.FunctionType(ir_return_type, ir_param_types)
#         func = llvmlite.ir.Function(module, function_signature, name=function_def.name)
#         i = 0
#         for arg in func.args:
#             python_arg = function_def.args.args[i]
#             i = i+1
#             substitutor[python_arg.arg] = arg
#
#         block = func.append_basic_block(name="entry")
#         builder = llvmlite.ir.IRBuilder(block)
#
#         if isinstance(function_def.body, typing.List):
#             for statement in function_def.body:
#                 to_process.append(statement)
#
#         while len(to_process) > 0:
#             python_node = to_process[0]
#             to_process.remove(python_node)
#             if isinstance(python_node, _ast.Assign):
#                 if len(python_node.targets) != 1:
#                     raise MLIRCompileError("tuple remaining in converted python")
#                 python_target = python_node.targets[0]
#                 if not isinstance(python_target, _ast.Name):
#                     raise MLIRCompileError("target must be a name")
#                 result_name = python_target.id
#                 # TODO: support more than mult
#                 python_rhs = python_node.value
#                 if not isinstance(python_rhs, _ast.BinOp):
#                     raise MLIRCompileError("Prototype only supports binary operations")
#                 op = python_rhs.op
#                 left = substitutor[python_rhs.left.id]
#                 right = substitutor[python_rhs.right.id]
#                 if not isinstance(python_rhs.right, _ast.Name) or not isinstance(python_rhs.left, _ast.Name):
#                     raise MLIRCompileError("no nesting allowed. Problem: " + python_target.id)
#                 if isinstance(op, _ast.Mult):
#                     if isinstance(left.type, llvm.ir.FloatType) and isinstance(right.type, llvm.ir.FloatType):
#                         instruction = builder.fmul(left, right, result_name)
#                     else:
#                         instruction = builder.mul(left, right, result_name)
#                 else:
#                     raise MLIRCompileError("I only compile multiplication for now")
#                 substitutor[result_name] = instruction
#             elif isinstance(python_node, _ast.Return):
#                 if not isinstance(python_node.value, _ast.Name):
#                     raise MLIRCompileError("no nesting allowed. Problem is return statement in " + function_def.name)
#                 instruction = substitutor[python_node.value.id]
#                 builder.ret(instruction)
#         return func