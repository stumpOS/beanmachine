//
// Created by Steffi Stumpos on 6/10/22.
//

#include <mlir/Target/LLVMIR/Export.h>
#include "MLIRBuilder.h"
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

#include "WorldTypeBuilder.h"
#include "pybind_utils.h"
#include "MLIRGenImpl.h"
#include "AbstractWorld.h"

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

using LogProbList = std::vector<paic_mlir::LogProbQueryTypes, std::allocator<paic_mlir::LogProbQueryTypes>>;
PYBIND11_MAKE_OPAQUE(LogProbList);

using Tensor = std::vector<float, std::allocator<float>>;
PYBIND11_MAKE_OPAQUE(Tensor);

PYBIND11_MODULE(paic_mlir, m) {
    m.doc() = "MVP for pybind module";
    paic_mlir::MLIRBuilder::bind(m);
    paic_mlir::Node::bind(m);
    py::enum_<paic_mlir::LogProbQueryTypes>(m, "LogProbQueryTypes")
            .value("TARGET_AND_CHILDREN", paic_mlir::LogProbQueryTypes::TARGET_AND_CHILDREN)
            .value("ALL_LATENT_VARIABLES", paic_mlir::LogProbQueryTypes::ALL_LATENT_VARIABLES)
            .value("TARGET", paic_mlir::LogProbQueryTypes::TARGET)
            .export_values();
    py::class_<paic_mlir::AbstractWorld, std::shared_ptr<paic_mlir::AbstractWorld>>(m, "AbstractWorld");
    py::class_<paic_mlir::WorldClassSpec>(m, "WorldClassSpec")
            .def(py::init<std::vector<paic_mlir::LogProbQueryTypes>>())
            .def("set_world_name", &paic_mlir::WorldClassSpec::set_world_name)
            .def("set_print_name", &paic_mlir::WorldClassSpec::set_print_name);
    bind_vector<paic_mlir::LogProbQueryTypes>(m, "LogProbQueryList");
    bind_vector<float>(m, "Tensor", true);
}

paic_mlir::InferenceFunctions::InferenceFunctions(pybind11::object contextObj) {}

void paic_mlir::MLIRBuilder::bind(py::module &m) {
    py::class_<MLIRBuilder>(m, "MLIRBuilder")
            .def(py::init<py::object>(), py::arg("context") = py::none())
            .def("to_metal", &MLIRBuilder::to_metal)
            .def("create_inference_functions", &MLIRBuilder::create_inference_functions);
    py::class_<InferenceFunctions, std::shared_ptr<InferenceFunctions>>(m, "InferenceFunctions")
            .def(py::init<py::object>(), py::arg("context") = py::none())
            .def("world_with_variables", &InferenceFunctions::create_world)
            .def("inference_function", &InferenceFunctions::inference_function);
}

paic_mlir::MLIRBuilder::MLIRBuilder(pybind11::object contextObj) {}

pybind11::float_ paic_mlir::MLIRBuilder::to_metal(std::shared_ptr<paic_mlir::PythonFunction> function, pybind11::float_ input) {
    // MLIR context (load any custom dialects you want to use)
    ::mlir::MLIRContext *context = new ::mlir::MLIRContext();
    context->loadDialect<mlir::func::FuncDialect>();
    context->loadDialect<mlir::math::MathDialect>();
    context->loadDialect<mlir::arith::ArithmeticDialect>();
    mlir::registerAllDialects(*context);

    // MLIR Module. Create the module
    std::vector<std::shared_ptr<PythonFunction>> functions{ function };
    std::shared_ptr<PythonModule> py_module = std::make_shared<PythonModule>(functions);
    MLIRGenImpl generator(*context);
    auto mlir_module = generator.generate_op(py_module);


    // todo: add passes and run module
    mlir::PassManager pm(context);
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createConvertMathToLLVMPass());
    pm.addPass(mlir::arith::createConvertArithmeticToLLVMPass());
    auto result = pm.run(mlir_module);

    // Lower to machine code
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    mlir::registerLLVMDialectTranslation(*context);

    // An optimization pipeline to use within the execution engine.
    auto optPipeline = mlir::makeOptimizingTransformer(
            /*optLevel=*/3, /*sizeLevel=*/0,
            /*targetMachine=*/nullptr);

    // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
    // the module.
    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.transformer = optPipeline;
    mlir_module->dump();
    engineOptions.llvmModuleBuilder = llvm::function_ref<std::unique_ptr<llvm::Module>(mlir::ModuleOp,llvm::LLVMContext &)>([](mlir::ModuleOp mod, llvm::LLVMContext &llvmContext){
        std::unique_ptr<llvm::Module> llvmModule = mlir::translateModuleToLLVMIR(mod, llvmContext);
        if (!llvmModule) {
            llvm::errs() << "Failed to emit LLVM IR\n";
        }
        bool BrokenDebugInfo = false;
        llvm::Module *module = llvmModule.get();
        if (llvm::verifyModule(*module, &llvm::errs(), &BrokenDebugInfo)){
            llvm::errs() << "LLVM IR invalid\n";
        }
        llvmModule->dump();
        return llvmModule;
    });
    auto maybeEngine = mlir::ExecutionEngine::create(mlir_module, engineOptions);

    assert(maybeEngine && "failed to construct an execution engine");
    auto &engine = maybeEngine.get();

    // Invoke the JIT-compiled function.
    float res = 0;
    auto invocationResult = engine->invoke(function->getName(), input.operator float(), mlir::ExecutionEngine::result(res));
    if (invocationResult) {
        llvm::errs() << "JIT invocation failed\n";
        throw 0;
    }
    return pybind11::float_(res);
}

std::shared_ptr<paic_mlir::InferenceFunctions> paic_mlir::MLIRBuilder::create_inference_functions(std::shared_ptr<paic_mlir::PythonFunction> function, const paic_mlir::WorldClassSpec &worldClassSpec) {
    std::string function_name = function->getName().data();
    std::vector<std::shared_ptr<PythonFunction>> functions{ function };
    std::shared_ptr<PythonModule> py_module = std::make_shared<PythonModule>(functions);

    py::cpp_function world_init = py::cpp_function([](std::shared_ptr<Tensor> v) {
        // should return type !llvm.ptr<struct<(i64, ptr<i8>)>>
        std::shared_ptr<paic_mlir::AbstractWorld> world = std::make_shared<paic_mlir::AbstractWorld>();
        world->_data = new double[v->size()];
        world->_size = v->size();
        for(int i=0;i<v->size();i++){
            world->_data[i] = v->at(i);
        }
        return world;
        }, py::arg("variables"));
    py::cpp_function inference_fnc;
    if(std::strcmp((function->getType().getName()).data(), "unit") == 0){
            inference_fnc = py::cpp_function([worldClassSpec, py_module, function_name](std::shared_ptr<paic_mlir::AbstractWorld> world) {
                // create a type using mlir
            ::mlir::MLIRContext *context = new ::mlir::MLIRContext();
            context->loadDialect<mlir::func::FuncDialect>();
            context->loadDialect<mlir::bm::BMDialect>();
            context->loadDialect<mlir::math::MathDialect>();
            context->loadDialect<mlir::arith::ArithmeticDialect>();
            context->loadDialect<mlir::memref::MemRefDialect>();
            mlir::registerAllDialects(*context);
            paic_mlir::WorldTypeBuilder builder(*context);
                std::map<std::string, std::vector<std::string>> external_types;
            std::vector<std::string> world_members;
            world_members.push_back("pointer");
            external_types[worldClassSpec.world_name()] = world_members;
            MLIRGenImpl generator(*context, external_types);
            mlir::ModuleOp mlir_module = generator.generate_op(py_module);
            mlir_module->dump();

            // todo: clean up passes. I don't think you need all these conversion passes; I think some can be nested
            mlir::PassManager pm(context);
            pm.addPass(mlir::bm::createLowerToFuncPass());
            pm.addPass(mlir::bm::createLowerToLLVMPass());
            auto result = pm.run(mlir_module);
    
            // Lower to machine code
            llvm::InitializeNativeTarget();
            llvm::InitializeNativeTargetAsmPrinter();
            mlir::registerLLVMDialectTranslation(*(mlir_module->getContext()));
            // disable optimizations (change first parameter to 3 to enable)
            auto optPipeline = mlir::makeOptimizingTransformer(0, 0, nullptr);

            // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
            // the module.
            mlir::ExecutionEngineOptions engineOptions;
            engineOptions.transformer = optPipeline;
            engineOptions.llvmModuleBuilder = llvm::function_ref<std::unique_ptr<llvm::Module>(mlir::ModuleOp,llvm::LLVMContext &)>([](mlir::ModuleOp mod, llvm::LLVMContext &llvmContext){
                std::unique_ptr<llvm::Module> llvmModule = mlir::translateModuleToLLVMIR(mod, llvmContext);
                if (!llvmModule) {
                    llvm::errs() << "Failed to emit LLVM IR\n";
                }
                llvmModule->dump();
                return llvmModule;
            });
            mlir_module->dump();
            auto maybeEngine = mlir::ExecutionEngine::create(mlir_module, engineOptions);
            assert(maybeEngine && "failed to construct an execution engine");
            auto &engine = maybeEngine.get();
            std::vector<double*> values(2);
            //values[0] = world->_data;
            values[1] = world->_data;

            auto ref = llvm::makeArrayRef(values);
            auto invocationResult = engine->invoke(function_name, ref, ref, 0, (int)world->_size, 1);
            if (invocationResult) {
                llvm::errs() << "JIT invocation failed\n";
            }
            delete[] world->_data;
        }, py::arg("world"));
    } else {
        llvm::errs() << "Only unit valued inference methods are supported atm\n";
    }

    std::shared_ptr<paic_mlir::InferenceFunctions> fncs = std::make_shared<paic_mlir::InferenceFunctions>();
    fncs->_create_world = world_init;
    fncs->_inference_function = inference_fnc;
    return fncs;
}