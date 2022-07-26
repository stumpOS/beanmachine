//
// Created by Steffi Stumpos on 7/15/22.
//
#include "MLIRBuilder.h"
#include <mlir/Target/LLVMIR/Export.h>
#include "bm/passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Host.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/MC/TargetRegistry.h"

#include "bm/BMDialect.h"
#include "pybind_utils.h"
#include <pybind11/stl_bind.h>
#include "llvm/MC/TargetRegistry.h"
//#include "llvm/ADT/Triple.h"

using Tensor = std::vector<float, std::allocator<float>>;
PYBIND11_MAKE_OPAQUE(Tensor);

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

// a world constant op should not be represented in python. Therefore do not expose world constant op as paic2 ast nodes.
// Instead, we will add the world const op separately.
// In order to do so, we only need the init data, world name, world element type, and world size
struct WorldLocalVar {
    WorldLocalVar(paic2::Location loc,std::string const& name, mlir::bm::WorldType const& wt):_loc(loc), _name(name), _worldType(wt){}
    paic2::Location _loc;
    std::string _name;
    mlir::bm::WorldType _worldType;
};

class MLIRGenImpl {
public:
    MLIRGenImpl(mlir::MLIRContext &context, paic2::WorldSpec const& spec) : builder(&context), _world_spec(spec){}
    mlir::ModuleOp generate_op(std::shared_ptr<paic2::PythonModule> pythonModule) {
        // We create an empty MLIR module and codegen functions one at a time and
        // add them to the module.
        theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

        for (std::shared_ptr<paic2::PythonFunction> f : pythonModule->getFunctions()){
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
    paic2::WorldSpec _world_spec;
    llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symbolTable;

    mlir::Location loc(const paic2::Location &loc) {
        return mlir::FileLineColLoc::get(builder.getStringAttr("file_not_implemented_yet"), loc.getLine(),
                                         loc.getCol());
    }

    mlir::Type typeForCode(paic2::PrimitiveCode code){
        switch(code){
            case paic2::PrimitiveCode::Float:
                return builder.getF32Type();
            case paic2::PrimitiveCode::Void:
                return builder.getNoneType();
            default:
                throw std::invalid_argument("unrecognized primitive type");
        }
    }

    mlir::bm::WorldType worldTypeFromDesc(paic2::WorldType* wt){
        std::vector<mlir::Type> members;
        mlir::Type elementType = typeForCode(wt->nodeType());
        ArrayRef<int64_t> shape(wt->length());
        auto dataType = mlir::RankedTensorType::get(shape, elementType);
        members.push_back(dataType);
        return mlir::bm::WorldType::get(members);
    }

    // There are two types supported: PrimitiveType and the builtin type WorldType
    // A World type is compiled into a composite type that contains an array, sets of function pointers, and metadata
    mlir::Type getType(std::shared_ptr<paic2::Type> type) {
        if(auto *var = dyn_cast<paic2::PrimitiveType>(type.get())){
            return typeForCode(var->code());
        } else if (auto *var = dyn_cast<paic2::WorldType>(type.get())) {
            return worldTypeFromDesc(var);
        }
        throw std::invalid_argument("unrecognized primitive type");
    }

    mlir::LogicalResult declare(llvm::StringRef var, mlir::Value value) {
        if (symbolTable.count(var))
            return mlir::failure();
        symbolTable.insert(var, value);
        return mlir::success();
    }

    mlir::Value mlirGen(paic2::GetValNode* expr) {
        if (auto variable = symbolTable.lookup(expr->getName()))
            return variable;

        emitError(loc(expr->loc()), "error: unknown variable '")
                << expr->getName() << "'";
        return nullptr;
    }

    mlir::LogicalResult mlirGen(paic2::ReturnNode* ret) {
        auto location = loc(ret->loc());
        mlir::Value expr = nullptr;
        if (ret->getValue()) {
            if (!(expr = mlirGen(ret->getValue().get())))
                return mlir::failure();
        }
        builder.create<mlir::bm::ReturnOp>(location, expr ? makeArrayRef(expr) : ArrayRef<mlir::Value>());
        return mlir::success();
    }

    mlir::Value mlirGen(paic2::CallNode* call) {
        llvm::StringRef callee = call->getCallee();
        auto location = loc(call->loc());

        // Codegen the operands first.
        SmallVector<mlir::Value, 4> operands;
        for (std::shared_ptr<paic2::Expression> expr : call->getArgs()) {
            auto arg = mlirGen(expr.get());
            if (!arg)
                return nullptr;
            operands.push_back(arg);
        }

        // TODO: make map and remove magic strings
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

        // TODO
        emitError(location, "User defined functions not supported yet");
        return nullptr;
    }

    mlir::OpState opFromCall(paic2::CallNode* call) {
        llvm::StringRef callee = call->getCallee();
        auto location = loc(call->loc());
        SmallVector<mlir::Value, 4> operands;
        for (std::shared_ptr<paic2::Expression> expr : call->getArgs()) {
            auto arg = mlirGen(expr.get());
            if (!arg)
                throw std::invalid_argument("invalid argument during expression generation");
            operands.push_back(arg);
        }

        mlir::Value receiver(nullptr);
        if(call->getReceiver().get() != nullptr){
            receiver = mlirGen(call->getReceiver().get());
        }
        if(strcmp(callee.data(), _world_spec.print_name().data()) == 0){
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

    mlir::Value mlirGen(paic2::Expression* expr) {
        switch (expr->getKind()) {
            case paic2::NodeKind::GetVal:
                return mlirGen(dynamic_cast<paic2::GetValNode*>(expr));
            case paic2::NodeKind::Constant:
            {
                auto primitive_type = dynamic_cast<paic2::PrimitiveType*>(expr->getType().get());
                if(primitive_type){
                    switch(primitive_type->code()){
                        case paic2::PrimitiveCode::Float:{
                            auto const_type = dynamic_cast<paic2::FloatConstNode*>(expr);
                            return builder.create<mlir::arith::ConstantFloatOp>(loc(expr->loc()), llvm::APFloat(const_type->getValue()), builder.getF32Type());
                        }
                        default:
                            throw std::invalid_argument("only float primitives are supported now");
                    }
                } else {
                    throw std::invalid_argument("only primitives are supported now");
                }
            }
            case paic2::NodeKind::Call:
                return mlirGen(dynamic_cast<paic2::CallNode*>(expr));
            default:
                emitError(loc(expr->loc()))
                        << "MLIR codegen encountered an unhandled expr kind '"
                        << Twine(expr->getKind()) << "'";
                return nullptr;
        }
    }

    mlir::Value mlirGen(paic2::VarNode* vardecl) {
        std::shared_ptr<paic2::Expression> init = vardecl->getInitVal();
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

    mlir::LogicalResult mlirGen(std::shared_ptr<paic2::BlockNode> blockNode, std::shared_ptr<WorldLocalVar> world_var) {
        ScopedHashTableScope<StringRef, mlir::Value> varScope(symbolTable);
        if(world_var != nullptr){
            // create world op
            auto dataType = world_var->_worldType.getElementTypes().front();
            auto dataAttribute = mlir::DenseElementsAttr::get(dataType, llvm::makeArrayRef(*(_world_spec.init_data)));
            mlir::TypeRange range({world_var->_worldType});
            std::vector<mlir::Attribute> attrElements;
            attrElements.push_back(dataAttribute);
            mlir::ArrayAttr dataAttr = builder.getArrayAttr(attrElements);
            auto world_op = builder.create<mlir::bm::WorldConstantOp>(loc(world_var->_loc), range, dataAttr);
            if(failed(declare(world_var->_name, world_op))){
                return mlir::failure();
            }
        }
        for (std::shared_ptr<paic2::Node> expr : blockNode->getChildren()) {
            if (auto *var = dyn_cast<paic2::VarNode>(expr.get())) {
                if (!mlirGen(var))
                    return mlir::failure();
                continue;
            }
            if (auto *var = dyn_cast<paic2::CallNode>(expr.get())) {
                if (!opFromCall(var))
                    return mlir::failure();
                continue;
            }
            if (auto *var = dyn_cast<paic2::ReturnNode>(expr.get())) {
                if (mlir::failed(mlirGen(var)))
                    return mlir::failure();
                continue;
            }
        }
        return mlir::success();
    }


    mlir::bm::FuncOp generate_op(std::shared_ptr<paic2::PythonFunction> &pythonFunction) {
        ScopedHashTableScope<llvm::StringRef, mlir::Value> varScope(symbolTable);
        builder.setInsertionPointToEnd(theModule.getBody());
        auto location = loc(pythonFunction->loc());

        std::vector<mlir::Type> arg_types;
        std::shared_ptr<WorldLocalVar> world_var = nullptr;
        for(std::shared_ptr<paic2::ParamNode> p : pythonFunction->getArgs()){
            auto py_type = p->getType();
            if(auto desc = dynamic_cast<paic2::WorldType*>(py_type.get())) {
                world_var = std::make_shared<WorldLocalVar>(p->loc(), p->getPyName(), worldTypeFromDesc(desc));
            } else {
                auto type = getType(p->getType());
                arg_types.push_back(type);
            }
        }

        mlir::TypeRange inputs(llvm::makeArrayRef(arg_types));
        mlir::FunctionType funcType;
        auto returnType = getType(pythonFunction->getType());
        if(returnType == nullptr || returnType.getTypeID() == builder.getNoneType().getTypeID()){
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

        // associate the parameter name in the python function with the block argument
        for (const auto nameValue : llvm::zip(protoArgs, entryBlock.getArguments())) {
            if (failed(declare(std::get<0>(nameValue)->getName(),std::get<1>(nameValue))))
                return nullptr;
        }

        builder.setInsertionPointToStart(&entryBlock);
        if (mlir::failed(mlirGen(pythonFunction->getBody(), world_var))) {
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

void paic2::MLIRBuilder::bind(py::module &m) {
    py::class_<paic2::WorldSpec>(m, "WorldSpec")
            .def(py::init())
            .def("set_print_name", &WorldSpec::set_print_name)
            .def("set_world_size", &WorldSpec::set_world_size)
            .def("set_world_name", &WorldSpec::set_world_name);

    py::class_<MLIRBuilder>(m, "MLIRBuilder")
            .def(py::init<py::object>(), py::arg("context") = py::none())
            .def("print_func_name", &MLIRBuilder::print_func_name)
            .def("infer", &MLIRBuilder::infer)
            .def("evaluate", &MLIRBuilder::evaluate);
    bind_vector<float>(m, "Tensor", true);
}

paic2::MLIRBuilder::MLIRBuilder(pybind11::object contextObj) {}

std::shared_ptr<mlir::ExecutionEngine> execution_engine_for_function(std::shared_ptr<paic2::PythonFunction> function, paic2::WorldSpec const& worldClassSpec) {
    // transform python to MLIR
    std::string function_name = function->getName().data();
    std::vector<std::shared_ptr<paic2::PythonFunction>> functions{ function };
    std::shared_ptr<paic2::PythonModule> py_module = std::make_shared<paic2::PythonModule>(functions);
    ::mlir::MLIRContext *context = new ::mlir::MLIRContext();
    context->loadDialect<mlir::func::FuncDialect>();
    context->loadDialect<mlir::bm::BMDialect>();
    context->loadDialect<mlir::math::MathDialect>();
    context->loadDialect<mlir::AffineDialect>();
    context->loadDialect<mlir::arith::ArithmeticDialect>();
    context->loadDialect<mlir::memref::MemRefDialect>();
    MLIRGenImpl generator(*context, worldClassSpec);
    mlir::ModuleOp mlir_module = generator.generate_op(py_module);
    mlir_module->dump();
    // lower to LLVM dialect
    mlir::PassManager pm(context);
    pm.addPass(mlir::bm::createLowerToFuncPass());
    pm.addPass(mlir::bm::createLowerToLLVMPass());
    auto result = pm.run(mlir_module);
    mlir_module->dump();

    // prepare environment for machine code generation
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    mlir::registerLLVMDialectTranslation(*(mlir_module->getContext()));

    auto optPipeline = mlir::makeOptimizingTransformer(
            /*optLevel=*/3, /*sizeLevel=*/0,
            /*targetMachine=*/nullptr);
    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.transformer = optPipeline;
    engineOptions.llvmModuleBuilder = llvm::function_ref<std::unique_ptr<llvm::Module>(mlir::ModuleOp,llvm::LLVMContext &)>([](mlir::ModuleOp mod, llvm::LLVMContext &llvmContext){
        std::unique_ptr<llvm::Module> llvmModule = mlir::translateModuleToLLVMIR(mod, llvmContext);
        if (!llvmModule) {
            llvm::errs() << "Failed to emit LLVM IR\n";
        }
        return llvmModule;
    });
    llvm::Expected<std::unique_ptr<mlir::ExecutionEngine>> maybeEngine = mlir::ExecutionEngine::create(mlir_module, engineOptions);
    assert(maybeEngine && "failed to construct an execution engine");
    auto &engine = maybeEngine.get();
    auto ptr = std::move(engine);
    return ptr;
}

void paic2::MLIRBuilder::print_func_name(std::shared_ptr<paic2::PythonFunction> function) {
    paic2::WorldSpec spec;
    spec.set_print_name("print");
    auto engine = execution_engine_for_function(function, spec);
}

void paic2::MLIRBuilder::infer(std::shared_ptr<paic2::PythonFunction> function, paic2::WorldSpec const& worldClassSpec, std::shared_ptr<std::vector<float>> init_nodes) {
    // Invoke the JIT-compiled function.
    WorldSpec spec(worldClassSpec);
    spec.init_data = init_nodes;
    auto engine = execution_engine_for_function(function, spec);
    float *data = new float[worldClassSpec.world_size()];
    for(int i=0;i<worldClassSpec.world_size();i++){
        float a = init_nodes->at(i);
        data[i] = a;
    }

    auto invocationResult = engine->invoke(function->getName().data());
    if (invocationResult) {
        llvm::errs() << "JIT invocation failed\n";
    }
}

pybind11::float_ paic2::MLIRBuilder::evaluate(std::shared_ptr<paic2::PythonFunction> function, pybind11::float_ input) {
    paic2::WorldSpec spec;
    auto engine = execution_engine_for_function(function, spec);

    // Invoke the JIT-compiled function.
    float res = 0;
    auto invocationResult = engine->invoke(function->getName(), input.operator float(), mlir::ExecutionEngine::result(res));
    if (invocationResult) {
        llvm::errs() << "JIT invocation failed\n";
        throw 0;
    }
    return pybind11::float_(res);
}
