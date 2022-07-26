//
// Created by Steffi Stumpos on 7/25/22.
//
#include "NoPyMLIRBuilder.h"
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
#include "llvm/MC/TargetRegistry.h"

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::makeArrayRef;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

namespace mlir {
    class MLIRContext;
    template <typename OpTy>
    class OwningOpRef;
    class ModuleOp;
} // namespace mlir

// a world constant op should not be represented in python. Therefore do not expose world constant op as nopy ast nodes.
// Instead, we will add the world const op separately.
// In order to do so, we only need the init data, world name, world element type, and world size
struct WorldLocalVar {
    WorldLocalVar(nopy::Location loc,std::string const& name, mlir::bm::WorldType const& wt):_loc(loc), _name(name), _worldType(wt){}
    nopy::Location _loc;
    std::string _name;
    mlir::bm::WorldType _worldType;
};

class MLIRGenImpl {
public:
    MLIRGenImpl(mlir::MLIRContext &context, nopy::WorldSpec const& spec) : builder(&context), _world_spec(spec){}
    mlir::ModuleOp generate_op(std::shared_ptr<nopy::PythonModule> pythonModule) {
        // We create an empty MLIR module and codegen functions one at a time and
        // add them to the module.
        theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

        for (std::shared_ptr<nopy::PythonFunction> f : pythonModule->getFunctions()){
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
    nopy::WorldSpec _world_spec;
    llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symbolTable;

    mlir::Location loc(const nopy::Location &loc) {
        return mlir::FileLineColLoc::get(builder.getStringAttr("file_not_implemented_yet"), loc.getLine(),
                                         loc.getCol());
    }

    mlir::Type typeForCode(nopy::PrimitiveCode code){
        switch(code){
            case nopy::PrimitiveCode::Float:
                return builder.getF64Type();
            case nopy::PrimitiveCode::Void:
                return builder.getNoneType();
            default:
                throw std::invalid_argument("unrecognized primitive type");
        }
    }

    mlir::bm::WorldType worldTypeFromDesc(nopy::WorldType* wt){
        std::vector<mlir::Type> members;
        mlir::Type elementType = typeForCode(wt->nodeType());
        ArrayRef<int64_t> shape(wt->length());
        auto dataType = mlir::RankedTensorType::get(shape, elementType);
        members.push_back(dataType);
        return mlir::bm::WorldType::get(members);
    }

    // There are two types supported: PrimitiveType and the builtin type WorldType
    // A World type is compiled into a composite type that contains an array, sets of function pointers, and metadata
    mlir::Type getType(std::shared_ptr<nopy::Type> type) {
        if(auto *var = dyn_cast<nopy::PrimitiveType>(type.get())){
            return typeForCode(var->code());
        } else if (auto *var = dyn_cast<nopy::WorldType>(type.get())) {
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

    mlir::Value mlirGen(nopy::GetValNode* expr) {
        if (auto variable = symbolTable.lookup(expr->getName()))
            return variable;

        emitError(loc(expr->loc()), "error: unknown variable '")
                << expr->getName() << "'";
        return nullptr;
    }

    mlir::LogicalResult mlirGen(nopy::ReturnNode* ret) {
        auto location = loc(ret->loc());
        mlir::Value expr = nullptr;
        if (ret->getValue()) {
            if (!(expr = mlirGen(ret->getValue().get())))
                return mlir::failure();
        }
        builder.create<mlir::bm::ReturnOp>(location, expr ? makeArrayRef(expr) : ArrayRef<mlir::Value>());
        return mlir::success();
    }

    mlir::Value mlirGen(nopy::CallNode* call) {
        llvm::StringRef callee = call->getCallee();
        auto location = loc(call->loc());

        // Codegen the operands first.
        SmallVector<mlir::Value, 4> operands;
        for (std::shared_ptr<nopy::Expression> expr : call->getArgs()) {
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

    mlir::OpState opFromCall(nopy::CallNode* call) {
        llvm::StringRef callee = call->getCallee();
        auto location = loc(call->loc());
        SmallVector<mlir::Value, 4> operands;
        for (std::shared_ptr<nopy::Expression> expr : call->getArgs()) {
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

    mlir::Value mlirGen(nopy::Expression* expr) {
        switch (expr->getKind()) {
            case nopy::NodeKind::GetVal:
                return mlirGen(dynamic_cast<nopy::GetValNode*>(expr));
            case nopy::NodeKind::Constant:
            {
                auto primitive_type = dynamic_cast<nopy::PrimitiveType*>(expr->getType().get());
                if(primitive_type){
                    switch(primitive_type->code()){
                        case nopy::PrimitiveCode::Float:{
                            auto const_type = dynamic_cast<nopy::FloatConstNode*>(expr);
                            return builder.create<mlir::arith::ConstantFloatOp>(loc(expr->loc()), llvm::APFloat(const_type->getValue()), builder.getF64Type());
                        }
                        default:
                            throw std::invalid_argument("only float primitives are supported now");
                    }
                } else {
                    throw std::invalid_argument("only primitives are supported now");
                }
            }
            case nopy::NodeKind::Call:
                return mlirGen(dynamic_cast<nopy::CallNode*>(expr));
            default:
                emitError(loc(expr->loc()))
                        << "MLIR codegen encountered an unhandled expr kind '"
                        << Twine(expr->getKind()) << "'";
                return nullptr;
        }
    }

    mlir::Value mlirGen(nopy::VarNode* vardecl) {
        std::shared_ptr<nopy::Expression> init = vardecl->getInitVal();
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

    mlir::LogicalResult mlirGen(std::shared_ptr<nopy::BlockNode> blockNode, std::shared_ptr<WorldLocalVar> world_var) {
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
        for (std::shared_ptr<nopy::Node> expr : blockNode->getChildren()) {
            if (auto *var = dyn_cast<nopy::VarNode>(expr.get())) {
                if (!mlirGen(var))
                    return mlir::failure();
                continue;
            }
            if (auto *var = dyn_cast<nopy::CallNode>(expr.get())) {
                if (!opFromCall(var))
                    return mlir::failure();
                continue;
            }
            if (auto *var = dyn_cast<nopy::ReturnNode>(expr.get())) {
                if (mlir::failed(mlirGen(var)))
                    return mlir::failure();
                continue;
            }
        }
        return mlir::success();
    }


    mlir::bm::FuncOp generate_op(std::shared_ptr<nopy::PythonFunction> &pythonFunction) {
        ScopedHashTableScope<llvm::StringRef, mlir::Value> varScope(symbolTable);
        builder.setInsertionPointToEnd(theModule.getBody());
        auto location = loc(pythonFunction->loc());

        std::vector<mlir::Type> arg_types;
        std::shared_ptr<WorldLocalVar> world_var = nullptr;
        for(std::shared_ptr<nopy::ParamNode> p : pythonFunction->getArgs()){
            auto py_type = p->getType();
            if(auto desc = dynamic_cast<nopy::WorldType*>(py_type.get())) {
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

nopy::NoPyMLIRBuilder::NoPyMLIRBuilder() {}

std::shared_ptr<mlir::ExecutionEngine> execution_engine_for_function(std::shared_ptr<nopy::PythonFunction> function, nopy::WorldSpec const& worldClassSpec) {
    // transform python to MLIR
    std::string function_name = function->getName().data();
    std::vector<std::shared_ptr<nopy::PythonFunction>> functions{ function };
    std::shared_ptr<nopy::PythonModule> py_module = std::make_shared<nopy::PythonModule>(functions);
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

void nopy::NoPyMLIRBuilder::print_func_name(std::shared_ptr<nopy::PythonFunction> function) {
    nopy::WorldSpec spec;
    spec.set_print_name("print");
    auto engine = execution_engine_for_function(function, spec);
}

void nopy::NoPyMLIRBuilder::infer(std::shared_ptr<nopy::PythonFunction> function, nopy::WorldSpec const& worldClassSpec, std::shared_ptr<std::vector<double>> init_nodes) {
    // Invoke the JIT-compiled function.
    nopy::WorldSpec spec(worldClassSpec);
    spec.init_data = init_nodes;
    auto engine = execution_engine_for_function(function, spec);
    double *data = new double[worldClassSpec.world_size()];
    for(int i=0;i<worldClassSpec.world_size();i++){
        float a = init_nodes->at(i);
        data[i] = a;
    }

    auto invocationResult = engine->invoke(function->getName().data());
    if (invocationResult) {
        llvm::errs() << "JIT invocation failed\n";
    }
    delete []data;
}
