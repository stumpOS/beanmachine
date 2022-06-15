//
// Created by Steffi Stumpos on 6/10/22.
//
#include <iostream>

#include "MLIRBuilder.h"
#include "ToyDialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/ScopedHashTable.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"
#include <numeric>

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::makeArrayRef;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

namespace py = pybind11;
namespace mlir {
    class MLIRContext;
    template <typename OpTy>
    class OwningOpRef;
    class ModuleOp;
} // namespace mlir

PYBIND11_MODULE(paic_mlir, m) {
    m.doc() = "MVP for pybind module";
    paic_mlir::MLIRBuilder::bind(m);
    paic_mlir::Node::bind(m);
}

void paic_mlir::MLIRBuilder::bind(py::module &m) {
    py::class_<MLIRBuilder>(m, "MLIRBuilder")
            .def(py::init<py::object>(), py::arg("context") = py::none())
            .def("to_metal", &MLIRBuilder::to_metal);
}

paic_mlir::MLIRBuilder::MLIRBuilder(pybind11::object contextObj) {}

namespace {
    class MLIRGenImpl {
    public:
        MLIRGenImpl(mlir::MLIRContext &context) : builder(&context){}

        mlir::ModuleOp generate_op(std::shared_ptr<paic_mlir::PythonModule> pythonModule) {
            // We create an empty MLIR module and codegen functions one at a time and
            // add them to the module.
            theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

            for (std::shared_ptr<paic_mlir::PythonFunction> f : pythonModule->getFunctions()){
                generate_op(f);
            }

            // Verify the module after we have finished constructing it, this will check
            // the structural properties of the IR and invoke any specific verifiers we
            // have on the Toy operations.
            if (failed(mlir::verify(theModule))) {
                theModule.emitError("module verification error");
                return nullptr;
            }
            return theModule;
        }
    private:
        mlir::ModuleOp theModule;
        mlir::OpBuilder builder;
        llvm::ScopedHashTable<StringRef, mlir::Value> symbolTable;
        llvm::ScopedHashTable<StringRef, mlir::func::FuncOp> functionSymbolTable;

        mlir::Location loc(const paic_mlir::Location &loc) {
            return mlir::FileLineColLoc::get(builder.getStringAttr("file_not_implemented_yet"), loc.getLine(),
                                             loc.getCol());
        }

        mlir::Type getTensorType(ArrayRef<int64_t> shape) {
            if (shape.empty())
                return mlir::UnrankedTensorType::get(builder.getF64Type());
            return mlir::RankedTensorType::get(shape, builder.getF64Type());
        }

        mlir::Type getType(const paic_mlir::Type type) {
            if(std::strcmp(type.getName().data(), "float") == 0){
                return builder.getF32Type();
            } else {
                // TODO: insert appropriate Not Implemented exception
                throw 0;
            }
        }

        mlir::LogicalResult declare(llvm::StringRef var, mlir::Value value) {
            if (symbolTable.count(var))
                return mlir::failure();
            symbolTable.insert(var, value);
            return mlir::success();
        }

        mlir::LogicalResult declare(llvm::StringRef var, mlir::func::FuncOp value) {
            if (functionSymbolTable.count(var))
                return mlir::failure();
            functionSymbolTable.insert(var, value);
            return mlir::success();
        }

        mlir::Value mlirGen(std::shared_ptr<paic_mlir::GetValNode> expr) {
            if (auto variable = symbolTable.lookup(expr->getName()))
                return variable;
            emitError(loc(expr->loc()), "error: unknown variable '")
                    << expr->getName() << "'";
            return nullptr;
        }

        mlir::Value mlirGen(paic_mlir::GetValNode* expr) {
            if (auto variable = symbolTable.lookup(expr->getName()))
                return variable;

            emitError(loc(expr->loc()), "error: unknown variable '")
                    << expr->getName() << "'";
            return nullptr;
        }

        mlir::LogicalResult mlirGen(paic_mlir::ReturnNode* ret) {
            auto location = loc(ret->loc());
            mlir::Value expr = nullptr;
            if (ret->getValue()) {
                if (!(expr = mlirGen(ret->getValue().get())))
                    return mlir::failure();
            }
            builder.create<mlir::func::ReturnOp>(location, expr ? makeArrayRef(expr) : ArrayRef<mlir::Value>());
            return mlir::success();
        }

        mlir::Value mlirGen(paic_mlir::CallNode* call) {
            llvm::StringRef callee = call->getCallee();
            auto location = loc(call->loc());

            // Codegen the operands first.
            SmallVector<mlir::Value, 4> operands;
            for (std::shared_ptr<paic_mlir::Expression> expr : call->getArgs()) {
                auto arg = mlirGen(expr.get());
                if (!arg)
                    return nullptr;
                operands.push_back(arg);
            }

            // Builtin calls have their custom operation, meaning this is a
            // straightforward emission.
            if (callee == "times") {
                if (call->getArgs().size() != 2) {
                    emitError(location, "MLIR codegen encountered an error: times "
                                        "accepts exactly two arguments");
                    return nullptr;
                }

                return builder.create<mlir::arith::MulFOp>(location, operands[0], operands[1]);
            }

            // Otherwise this is a call to a user-defined function. Calls to
            // user-defined functions are mapped to a custom call that takes the callee
            // name as an attribute.
            // arguments: FuncOp callee, ValueRange operands = {}
            if (mlir::func::FuncOp functionOp = functionSymbolTable.lookup(callee)){
                auto call = builder.create<mlir::func::CallOp>(location, functionOp, operands);
                throw 0;
                // TODO: discover correct abstraction for values
             //   return call;
            } else {
                emitError(location, "MLIR codegen encountered an error: toy.transpose "
                                    "does not accept multiple arguments");
                return nullptr;
            }

        }

        mlir::Value mlirGen(paic_mlir::ConstNode<float>* expr) {
            auto type = getType(expr->getType());
            return builder.create<mlir::arith::ConstantFloatOp>(loc(expr->loc()), llvm::APFloat(expr->getValue()), mlir::FloatType());
        }

        mlir::Value mlirGen(paic_mlir::Expression* expr) {
            switch (expr->getKind()) {
                case paic_mlir::NodeKind::GetVal:
                    return mlirGen(dynamic_cast<paic_mlir::GetValNode*>(expr));
                case paic_mlir::NodeKind::Constant:
                    // TODO: cast to ConstNode parent and query primitive type
                    return mlirGen(dynamic_cast<paic_mlir::ConstNode<float>*>(expr));
                case paic_mlir::NodeKind::Call:
                    return mlirGen(dynamic_cast<paic_mlir::CallNode*>(expr));
                default:
                    emitError(loc(expr->loc()))
                            << "MLIR codegen encountered an unhandled expr kind '"
                            << Twine(expr->getKind()) << "'";
                    return nullptr;
            }
        }
        mlir::Value mlirGen(paic_mlir::VarNode* vardecl) {
            std::shared_ptr<paic_mlir::Expression> init = vardecl->getInitVal();
            if (!init) {
                emitError(loc(vardecl->loc()),"missing initializer in variable declaration");
                return nullptr;
            }
            mlir::Value value = mlirGen(init.get());
            if (!value)
                return nullptr;
            if (failed(declare(vardecl->getName(), value)))
                return nullptr;
            return value;
        }
        /// Codegen a list of expression, return failure if one of them hit an error.
        mlir::LogicalResult mlirGen(std::shared_ptr<paic_mlir::BlockNode> blockNode) {
            ScopedHashTableScope<StringRef, mlir::Value> varScope(symbolTable);
            for (std::shared_ptr<paic_mlir::Node> expr : blockNode->getChildren()) {
                // Specific handling for variable declarations, return statement, and
                // print. These can only appear in block list and not in nested
                // expressions.
                if (auto *var = dyn_cast<paic_mlir::VarNode>(expr.get())) {
                    if (!mlirGen(var))
                        return mlir::failure();
                    continue;
                }
                if (auto *var = dyn_cast<paic_mlir::ReturnNode>(expr.get())) {
                    if (mlir::failed(mlirGen(var)))
                        return mlir::failure();
                    continue;
                }
            }
            return mlir::success();
        }

        mlir::func::FuncOp generate_op(std::shared_ptr<paic_mlir::PythonFunction> &pythonFunction) {
            // Create a scope in the symbol table to hold variable declarations.
            ScopedHashTableScope<llvm::StringRef, mlir::Value> varScope(symbolTable);

            // Create an MLIR function for the given prototype.
            builder.setInsertionPointToEnd(theModule.getBody());
            // create FuncOp
            auto location = loc(pythonFunction->loc());

            // TODO: change the number of inlined elements in small array from 4?
            int i=0;
            std::vector<mlir::Type> arg_types(pythonFunction->getArgs().size());
            for(std::shared_ptr<paic_mlir::ParamNode> p : pythonFunction->getArgs()){
                auto type = getType(p->getType());
                arg_types[i++] = type;
            }

            // create a function using the Func dialect
            mlir::TypeRange inputs(llvm::makeArrayRef(arg_types));
            mlir::FunctionType funcType = builder.getFunctionType(inputs, getType(pythonFunction->getType()));
            // TODO: add attributes here if relevant
            mlir::func::FuncOp func_op = builder.create<mlir::func::FuncOp>(location, pythonFunction->getName(), funcType);

            func_op.addEntryBlock();
            mlir::Block &entryBlock = func_op.front();
            auto protoArgs = pythonFunction->getArgs();

            // Declare all the function arguments in the symbol table.
            for (const auto nameValue :
                    llvm::zip(protoArgs, entryBlock.getArguments())) {
                if (failed(declare(std::get<0>(nameValue)->getName(),
                                   std::get<1>(nameValue))))
                    return nullptr;
            }
            // Set the insertion point in the builder to the beginning of the function
            // body, it will be used throughout the codegen to create operations in this
            // function.
            builder.setInsertionPointToStart(&entryBlock);
            // Emit the body of the function.
            if (mlir::failed(mlirGen(pythonFunction->getBody()))) {
                func_op.erase();
                return nullptr;
            }
            mlir::func::ReturnOp returnOp;
            if (!entryBlock.empty())
                returnOp = dyn_cast<mlir::func::ReturnOp>(entryBlock.back());
            if (!returnOp) {
                builder.create<mlir::func::ReturnOp>(loc(pythonFunction->loc()));
            } else if (!returnOp.operands().empty()) {
                // Otherwise, if this return operation has an operand then add a result to
                // the function.
                func_op.setType(builder.getFunctionType(func_op.getFunctionType().getInputs(), getType(pythonFunction->getType())));
            }
            return func_op;
        }
    };
}

pybind11::object paic_mlir::MLIRBuilder::to_metal(std::shared_ptr<paic_mlir::PythonFunction> function) {
    // TODO: if not already compiled, compile the Python function by
    //      mapping to mlir and compiling. Call compiled function and return results.
    // TODO: accept the functions relevant to the input model. Generate implementation of the ModelWorld

    std::cout << function->getName().data();
    // from the Python function, create MLIR

    // MLIR context (load any custom dialects you want to use)
    ::mlir::MLIRContext *context = new ::mlir::MLIRContext();
    context->loadDialect<mlir::toy::ToyDialect>();
    context->loadDialect<mlir::func::FuncDialect>();
    context->loadDialect<mlir::math::MathDialect>();
    context->loadDialect<mlir::arith::ArithmeticDialect>();
    mlir::registerAllDialects(*context);

    // MLIR Module. Create the module
    std::vector<std::shared_ptr<PythonFunction>> functions{ function };
    std::shared_ptr<PythonModule> py_module = std::make_shared<PythonModule>(functions);
    MLIRGenImpl generator(*context);
    auto mlir_module = generator.generate_op(py_module);


    throw 0;
}

// TODO: