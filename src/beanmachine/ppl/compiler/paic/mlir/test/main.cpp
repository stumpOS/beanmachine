#include <iostream>
#include <string>
#include "ToyDialect.h"
#include "Parser.h"
#include "Lexer.h"
#include "ToyAST.h"
#include "Passes.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "MLIRGenerator.h"
namespace cl = llvm::cl;

std::unique_ptr<demo::ModuleAST> parseInputFile(llvm::StringRef filename) {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
            llvm::MemoryBuffer::getFileOrSTDIN(filename);
    if (std::error_code ec = fileOrErr.getError()) {
        llvm::errs() << "Could not open input file: " << ec.message() << "\n";
        return nullptr;
    }
    auto buffer = fileOrErr.get()->getBuffer();
    demo::LexerBuffer lexer(buffer.begin(), buffer.end(), std::string(filename));
    demo::Parser parser(lexer);
    return parser.parseModule();
}

void print_ast(std::ostream &os, std::string filename) {
    auto moduleAST = parseInputFile(filename);
    demo::dump(*moduleAST);
}

void print_mlir(std::ostream &os, std::string filename, bool should_opt, bool lower_to_affine){
    auto ast = parseInputFile(filename);
    ::mlir::MLIRContext *context = new ::mlir::MLIRContext();
    context->loadDialect<mlir::toy::ToyDialect>();

    auto module = demo::mlirGen(*context, *ast);
    if(should_opt){
        mlir::PassManager pm(context);
        pm.addPass(mlir::createInlinerPass());
        mlir::OpPassManager &optPM = pm.nest<mlir::toy::FuncOp>();
        optPM.addPass(mlir::toy::createShapeInferencePass());
        optPM.addPass(mlir::createCanonicalizerPass());
        optPM.addPass(mlir::createCSEPass());
        if(lower_to_affine){
            // Partially lower the toy dialect.
            pm.addPass(mlir::toy::createLowerToAffinePass());

            // Add a few cleanups post lowering.
            mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
            optPM.addPass(mlir::createCanonicalizerPass());
            optPM.addPass(mlir::createCSEPass());

            // Add optimizations that are possible now that we have an affine transform
            optPM.addPass(mlir::createLoopFusionPass());
            optPM.addPass(mlir::createAffineScalarReplacementPass());
        }
        pm.run(*module);
    }
    module->dump();
    delete context;
}

int execute_mlir(std::ostream &os, std::string filename){
    auto ast = parseInputFile(filename);
    ::mlir::MLIRContext *context = new ::mlir::MLIRContext();
    context->loadDialect<mlir::toy::ToyDialect>();

    auto module = demo::mlirGen(*context, *ast);
    mlir::PassManager pm(context);
    pm.addPass(mlir::createInlinerPass());
    mlir::OpPassManager &optPM = pm.nest<mlir::toy::FuncOp>();
    optPM.addPass(mlir::toy::createShapeInferencePass());
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());
    // Partially lower the toy dialect.
    pm.addPass(mlir::toy::createLowerToAffinePass());

    // Add a few cleanups post lowering.
    mlir::OpPassManager &optPM2 = pm.nest<mlir::func::FuncOp>();
    optPM2.addPass(mlir::createCanonicalizerPass());
    optPM2.addPass(mlir::createCSEPass());

    // Add optimizations that are possible now that we have an affine transform
    optPM2.addPass(mlir::createLoopFusionPass());
    optPM2.addPass(mlir::createAffineScalarReplacementPass());
    pm.addPass(mlir::toy::createLowerToLLVMPass());
    pm.run(*module);

    // lower to llvm ir and execute
    // Initialize LLVM targets.
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    // Register the translation from MLIR to LLVM IR, which must happen before we
    // can JIT-compile.
    //  std::unique_ptr<llvm::Module> llvmModule = mlir::translateModuleToLLVMIR(module);
    mlir::registerLLVMDialectTranslation(*module->getContext());

    // An optimization pipeline to use within the execution engine.
    auto optPipeline = mlir::makeOptimizingTransformer(
            /*optLevel=*/3, /*sizeLevel=*/0,
            /*targetMachine=*/nullptr);

    // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
    // the module.
    module->dump();
    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.transformer = optPipeline;
//    engineOptions.llvmModuleBuilder = llvm::function_ref<std::unique_ptr<llvm::Module>(mlir::ModuleOp,llvm::LLVMContext &)>([](mlir::ModuleOp mod, llvm::LLVMContext &llvmContext){
//        std::unique_ptr<llvm::Module> llvmModule = mlir::translateModuleToLLVMIR(mod, llvmContext);
//        if (!llvmModule) {
//            llvm::errs() << "Failed to emit LLVM IR\n";
//        }
//
//        llvmModule->dump();
//        return llvmModule;
//    });
    auto maybeEngine = mlir::ExecutionEngine::create(*module, engineOptions);
    assert(maybeEngine && "failed to construct an execution engine");
    auto &engine = maybeEngine.get();

    // Invoke the JIT-compiled function.
    auto invocationResult = engine->invokePacked("main");
    if (invocationResult) {
        llvm::errs() << "JIT invocation failed\n";
        return -1;
    }

    return 0;
}

int main(int argc, char *argv[]) {
    std::string root(reinterpret_cast<const char *>(argv[argc - 1]));
    auto directory = root.substr(root.find_first_of('=')+1, root.size());
    std::string test_case(directory + "/test/test.toy");
    char a = '0';
    std::cout << "> ";
    while(a != 'q'){
        std::cout << "Enter 'a' to print AST,\n 'q' to quit,\n 'c' to compile,\n 'o' to emit mlir with optimization passes \n 'm' to emit mlir with no optimization passes \n 'p' to emit mlir lowered to affine dialect" << std::endl;
        std::cout << "> ";
        std::cin >> a;
        switch(a){
            case 'c':
                execute_mlir(std::cout, test_case);
                break;
            case 'a':
                print_ast(std::cout, test_case);
                break;
            case 'o' :
                print_mlir(std::cout, test_case, true, false);
                break;
            case 'p' :
                print_mlir(std::cout, test_case, true, true);
                break;
            case 'm' :
                print_mlir(std::cout, test_case, false, false);
                break;
        }
        std::cout << "> ";
    }
    return 0;
}
