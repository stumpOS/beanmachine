//
// Created by Steffi Stumpos on 6/27/22.
//

#ifndef PAIC_IR_MLIRGENIMPL_H
#define PAIC_IR_MLIRGENIMPL_H
#include "MLIRBuilder.h"
#include "bm/BMDialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRefOps.h.inc"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectInterface.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/IR/Module.h"
#include "WorldTypeBuilder.h"
#include "pybind_utils.h"
#include <string>
#include <map>
#include<stdexcept>

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

namespace {
    class MLIRGenImpl {
    public:
        MLIRGenImpl(mlir::MLIRContext &context) : builder(&context){}
        MLIRGenImpl(mlir::MLIRContext &context, std::map<std::string, std::vector<std::string>> const& external_types) : builder(&context), _external_types(external_types){ }

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
        std::map<std::string, std::vector<std::string>> _external_types;
        mlir::ModuleOp theModule;
        mlir::OpBuilder builder;
        llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symbolTable;
        llvm::ScopedHashTable<llvm::StringRef, mlir::func::FuncOp> functionSymbolTable;

        mlir::Location loc(const paic_mlir::Location &loc) {
            return mlir::FileLineColLoc::get(builder.getStringAttr("file_not_implemented_yet"), loc.getLine(),
                                             loc.getCol());
        }

        mlir::Type getType(const char* type_name) {
            // TODO: should "unit" type be mapped to None built in type?
            // TODO: refactor into map
            if(std::strcmp(type_name, "float") == 0){
                return builder.getF32Type();
            }
            // TODO: delete me in favor of an llvm lowering
            else if (std::strcmp(type_name, "pointer") == 0) {
                // What does this mean? A: the interpretation is up to the backend (e.g. LLVM)
                mlir::Attribute memSpace = mlir::IntegerAttr::get(mlir::IntegerType::get(builder.getContext(), 64), 7);
                mlir::ShapedType unrankedTensorType = mlir::UnrankedMemRefType::get(builder.getF32Type(), memSpace);
                return unrankedTensorType;
            } else {
                auto member_types_maybe = _external_types.find(type_name);
                // TODO: protect against recursion
                if (member_types_maybe != _external_types.end()) {
                    std::vector<mlir::Type> members;
                    std::vector<std::string> type_names = member_types_maybe->second;
                    for (std::string member_type_name: type_names) {
                        members.push_back(getType(member_type_name.data()));
                    }
                    mlir::Type structType = mlir::bm::WorldType::get(members);
                    return structType;
                } else {
                    return nullptr;
                }
            }
        }

        // There are two types supported:
        mlir::Type getType(const paic_mlir::Type type) {
            const char* type_name = type.getName().data();
            return getType(type_name);
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
            builder.create<mlir::bm::ReturnOp>(location, expr ? makeArrayRef(expr) : ArrayRef<mlir::Value>());
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
            // TODO: make map
            if (callee == "times") {
                if (call->getArgs().size() != 2) {
                    emitError(location, "MLIR codegen encountered an error: times "
                                        "accepts exactly two arguments");
                    return nullptr;
                }
                return builder.create<mlir::arith::MulFOp>(location, operands[0], operands[1]);
            } else if (callee == "math.pow") {
                return builder.create<mlir::math::PowFOp>(location, operands[0], operands[1]);
            }

            // Otherwise this is a call to a user-defined function. Calls to
            // user-defined functions are mapped to a custom call that takes the callee
            // name as an attribute.
            // arguments: FuncOp callee, ValueRange operands = {}
            if (mlir::func::FuncOp functionOp = functionSymbolTable.lookup(callee)){
                auto call = builder.create<mlir::func::CallOp>(location, functionOp, operands);
                // TODO: discover correct abstraction for values
                emitError(location, "User defined functions not supported yet");
                return nullptr;
            } else {
                emitError(location, "Unreconized function :" + callee);
                return nullptr;
            }
        }

        mlir::OpState mlirGen_Expression(paic_mlir::CallNode* call) {
            llvm::StringRef callee = call->getCallee();
            auto location = loc(call->loc());

            // Codegen the operands first.
            SmallVector<mlir::Value, 4> operands;
            for (std::shared_ptr<paic_mlir::Expression> expr : call->getArgs()) {
                auto arg = mlirGen(expr.get());
                if (!arg)
                    throw std::invalid_argument("invalid argument during expression generation");
                operands.push_back(arg);
            }

            mlir::Value receiver(nullptr);
            if(call->getReceiver().get() != nullptr){
                receiver = mlirGen(call->getReceiver().get());
            }
            if(strcmp(callee.data(), "print") == 0){
                // we add an operator here so that all the operations relevant to printing are kept
                // inside a single operation until we are ready to lower to llvm ir
                if(receiver != nullptr && receiver.getType().isa<mlir::bm::WorldType>()){
                    mlir::bm::PrintWorldOp result =  builder.create<mlir::bm::PrintWorldOp>(location, receiver);
                    return result;
                } else {
                    throw std::invalid_argument("only world print operations are supported");
                }
            } else {
                throw std::invalid_argument("only world print operations are supported");
            }
        }

        mlir::Value mlirGen(paic_mlir::ConstNode<float>* expr) {
            return builder.create<mlir::arith::ConstantFloatOp>(loc(expr->loc()), llvm::APFloat(expr->getValue()), builder.getF32Type());
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
//                if (auto *var = dyn_cast<paic_mlir::CallNode>(expr.get())) {
//                    if (!mlirGen_Expression(var))
//                        return mlir::failure();
//                    continue;
//                }
                if (auto *var = dyn_cast<paic_mlir::ReturnNode>(expr.get())) {
                    if (mlir::failed(mlirGen(var)))
                        return mlir::failure();
                    continue;
                }
            }
            return mlir::success();
        }

       mlir::bm::FuncOp generate_op(std::shared_ptr<paic_mlir::PythonFunction> &pythonFunction) {
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

           mlir::FunctionType funcType;
           auto returnType = getType(pythonFunction->getType());
           if(returnType == nullptr){
               // TODO: error out here
               funcType = builder.getFunctionType(inputs, {});
           } else {
               funcType = builder.getFunctionType(inputs, returnType);
           }
           mlir::bm::FuncOp func_op = builder.create<mlir::bm::FuncOp>(location, pythonFunction->getName(), funcType);
           func_op->setAttr("llvm.emit_c_interface", mlir::UnitAttr::get(func_op->getContext()));
           if(func_op.empty()){
               func_op.addEntryBlock();
           }

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
           mlir::bm::ReturnOp returnOp;
           if (!entryBlock.empty())
               returnOp = dyn_cast<mlir::bm::ReturnOp>(entryBlock.back());
           if (!returnOp) {
               builder.create<mlir::bm::ReturnOp>(loc(pythonFunction->loc()));
           } else if (!returnOp.hasOperand()) {
               // Otherwise, if this return operation has an operand then add a result to
               // the function.
               auto returnType = getType(pythonFunction->getType());
               if(returnType == nullptr){
                   func_op.setType(builder.getFunctionType(func_op.getFunctionType().getInputs(), {}));
               } else {
                   func_op.setType(builder.getFunctionType(func_op.getFunctionType().getInputs(), returnType));
               }

           }
           return func_op;
       }
    };
}
#endif //PAIC_IR_MLIRGENIMPL_H
