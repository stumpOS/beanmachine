//
// Created by Steffi Stumpos on 7/25/22.
//
#include <mlir/Target/LLVMIR/Export.h>
#include "bm/BMDialect.h"
#include "bm/passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Host.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/IR/Verifier.h"
#include <iostream>
#include <vector>
#include "NoPyMLIRBuilder.h"

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::makeArrayRef;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

int main(int argc, char *argv[]) {
    nopy::Location loc(0,0);
    auto unitType = std::make_shared<nopy::PrimitiveType>(nopy::PrimitiveCode::Void);
    auto worldType = std::make_shared<nopy::WorldType>(nopy::PrimitiveCode::Float, 3);
    std::shared_ptr<nopy::Type> base = unitType;

    std::shared_ptr<std::vector<std::shared_ptr<nopy::ParamNode>>> params = std::make_shared<std::vector<std::shared_ptr<nopy::ParamNode>>>();
    std::shared_ptr<nopy::Type> base2 = worldType;
    std::shared_ptr<nopy::ParamNode> w = std::make_shared<nopy::ParamNode>(loc, "world", base2);
    params->push_back(w);

    std::vector<std::shared_ptr<nopy::Expression>> args;
    nopy::WorldSpec spec;
    spec.set_print_name("print");
    std::shared_ptr<nopy::GetValNode> gv = std::make_shared<nopy::GetValNode>(loc, w->getPyName(), base2);
    std::shared_ptr<nopy::CallNode> cn = std::make_shared<nopy::CallNode>(loc, spec.print_name(), args, gv, base);

    std::vector<std::shared_ptr<nopy::Node>> statements;
    statements.push_back(cn);
    std::shared_ptr<nopy::BlockNode> block = std::make_shared<nopy::BlockNode>(loc, statements);
    auto pythonFunc =  std::make_shared<nopy::PythonFunction>(loc, "foo", base, *params, block);
    nopy::NoPyMLIRBuilder builder;
    std::shared_ptr<std::vector<double>> init_nodes = std::make_shared<std::vector<double>>();
    spec.set_world_size(3);
    for(int i=0;i<3;i++){
        init_nodes->push_back(0.3f + (double)i);
    }
    builder.infer(pythonFunc, spec, init_nodes);

//    std::string function_name = "foo";
//
//    ::mlir::MLIRContext *context = new ::mlir::MLIRContext();
//    context->loadDialect<mlir::func::FuncDialect>();
//    context->loadDialect<mlir::bm::BMDialect>();
//    context->loadDialect<mlir::math::MathDialect>();
//    context->loadDialect<mlir::arith::ArithmeticDialect>();
//    context->loadDialect<mlir::memref::MemRefDialect>();
//    mlir::registerAllDialects(*context);
//    // create mlir_module
//    mlir::OpBuilder builder(context);
//    mlir::ModuleOp mlir_module = mlir::ModuleOp::create(builder.getUnknownLoc());
//    mlir_module->dump();
//
//    // create the function
//    llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symbolTable;
//    ScopedHashTableScope<llvm::StringRef, mlir::Value> varScope(symbolTable);
//
//    // Create an MLIR function for the given prototype.
//    builder.setInsertionPointToEnd(mlir_module.getBody());
//    // create FuncOp
//    auto location = mlir::FileLineColLoc::get(builder.getStringAttr("file_not_implemented_yet"), 0, 0);
//
//    // TODO: change the number of inlined elements in small array from 4?
//    int i=0;
//    std::vector<mlir::Type> arg_types(1);
//    std::vector<mlir::Type> members;
//    mlir::Attribute memSpace = mlir::IntegerAttr::get(mlir::IntegerType::get(builder.getContext(), 64), 7);
//    mlir::ShapedType unrankedTensorType = mlir::UnrankedMemRefType::get(builder.getF32Type(), memSpace);
//    members.push_back(unrankedTensorType);
//    mlir::Type structType = mlir::bm::WorldType::get(members);
//    arg_types[0] = structType;
//
//    // create a function using the Func dialect
//    mlir::TypeRange inputs(llvm::makeArrayRef(arg_types));
//
//    mlir::FunctionType funcType = builder.getFunctionType(inputs, {});
//    mlir::bm::FuncOp func_op = builder.create<mlir::bm::FuncOp>(location, function_name, funcType);
//    func_op->setAttr("llvm.emit_c_interface", mlir::UnitAttr::get(func_op->getContext()));
//    if(func_op.empty()){
//        func_op.addEntryBlock();
//    }
//
//    mlir::Block &entryBlock = func_op.front();
//    // Set the insertion point in the builder to the beginning of the function
//    // body, it will be used throughout the codegen to create operations in this
//    // function.
//    builder.setInsertionPointToStart(&entryBlock);
//    // Emit the body of the function. Add a print op
//    mlir::bm::PrintWorldOp result =  builder.create<mlir::bm::PrintWorldOp>(location, func_op.front().getArgument(0));
//    builder.create<mlir::bm::ReturnOp>(location);
//
//    mlir::PassManager pm(context);
//    pm.addPass(mlir::bm::createLowerToFuncPass());
//    pm.addPass(mlir::bm::createLowerToLLVMPass());
//    auto result_of_run = pm.run(mlir_module);
//
//    // Lower to machine code
//    llvm::InitializeNativeTarget();
//    llvm::InitializeNativeTargetAsmPrinter();
//    mlir::registerLLVMDialectTranslation(*(mlir_module->getContext()));
//    // disable optimizations (change first parameter to 3 to enable)
//    auto optPipeline = mlir::makeOptimizingTransformer(0, 0, nullptr);
//
//    // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
//    // the module.
//    mlir::ExecutionEngineOptions engineOptions;
//    engineOptions.transformer = optPipeline;
//    engineOptions.llvmModuleBuilder = llvm::function_ref<std::unique_ptr<llvm::Module>(mlir::ModuleOp,llvm::LLVMContext &)>([](mlir::ModuleOp mod, llvm::LLVMContext &llvmContext){
//        std::unique_ptr<llvm::Module> llvmModule = mlir::translateModuleToLLVMIR(mod, llvmContext);
//        if (!llvmModule) {
//            llvm::errs() << "Failed to emit LLVM IR\n";
//        }
//        bool BrokenDebugInfo = false;
//        llvm::Module *module = llvmModule.get();
//        if (llvm::verifyModule(*module, &llvm::errs(), &BrokenDebugInfo)){
//            llvm::errs() << "LLVM IR invalid\n";
//        }
//        //llvmModule->dump();
//        return llvmModule;
//    });
//
//    mlir_module->dump();
//    auto maybeEngine = mlir::ExecutionEngine::create(mlir_module, engineOptions);
//    assert(maybeEngine && "failed to construct an execution engine");
//    auto &engine = maybeEngine.get();
//
//    std::vector<double*> values;
//    for(int i=0;i<5;i++){
//        double* array = new double[5];
//        for(int j=0;j<5;j++){
//            array[j] = (double)(i + j+0.9);
//        }
//        values.push_back(array);
//    }
//    llvm::ArrayRef<double*> ref2 = llvm::makeArrayRef(values);
//    auto invocationResult = engine->invoke(function_name, ref2, ref2, 0, 5, 0);
//    if (invocationResult) {
//        llvm::errs() << "JIT invocation failed\n";
//    }
//    for(int i=0;i<5;i++){
//        delete[] values[i];
//    }
}