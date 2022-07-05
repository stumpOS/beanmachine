//
// Created by Steffi Stumpos on 6/24/22.
//
#include "WorldTypeBuilder.h"
#include "mlir/IR/Verifier.h"
#include "bm/BMDialect.h"

paic_mlir::WorldTypeBuilder::WorldTypeBuilder(mlir::MLIRContext &context): builder(&context) {

}
mlir::Location paic_mlir::WorldTypeBuilder::generated_loc() {
    return mlir::FileLineColLoc::get(builder.getStringAttr("file_not_implemented_yet"), 0,
                                     0);
}

/// Emit a constant for a struct literal. It will be emitted as an array of
/// other literals in an Attribute attached to a `toy.struct_constant`
/// operation. This function returns the generated constant, along with the
/// corresponding struct type.
std::pair<mlir::ArrayAttr, mlir::Type> paic_mlir::WorldTypeBuilder::getConstantAttr(std::vector<float> &values) {
    std::vector<mlir::Attribute> attrElements;
    std::vector<mlir::Type> typeElements;

    for (auto var : values) {
        mlir::Type elementType = builder.getF64Type();
        auto dataType = mlir::RankedTensorType::get({}, elementType);

        // This is the actual attribute that holds the list of values for this
        // tensor literal.
        mlir::DenseElementsAttr attribute = mlir::DenseElementsAttr::get(dataType, llvm::makeArrayRef(var));
        attrElements.push_back(attribute);
        typeElements.push_back(mlir::UnrankedTensorType::get(builder.getF64Type()));
    }
    mlir::ArrayAttr dataAttr = builder.getArrayAttr(attrElements);
    // TODO: The builder should not have to specify data layouts for the world type; it is a built in type
    mlir::Type dataType = mlir::bm::WorldType::get(typeElements);
    return std::make_pair(dataAttr, dataType);
}

// say the world class spec is `TARGET`.
// then we want to create a struct that has the following data:
// (1) std::shared_ptr<std::vector<std::function<float(float, std::shared_ptr<std::vector<float>>)>>> _log_prob_me_functions;
// (2) std::shared_ptr<std::vector<float>> _variables;
mlir::ModuleOp paic_mlir::WorldTypeBuilder::generate_op(const paic_mlir::WorldClassSpec &worldClassSpec) {
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

    // Create an instance of that struct type by materializing a constant
    std::vector<double> values;
    for(int i=0;i<5;i++){
        values.push_back((double)i);
    }

    std::vector<int64_t> shape;
    shape.push_back(5);
    auto tensor_type = mlir::RankedTensorType::get(shape, builder.getF64Type());
    mlir::DenseElementsAttr attribute = mlir::DenseElementsAttr::get(tensor_type, llvm::makeArrayRef(values));
    mlir::bm::ConstantOp op = builder.create<mlir::bm::ConstantOp>(generated_loc(), mlir::UnrankedTensorType::get(builder.getF64Type()), attribute);

    // Create a struct type that has a single unranked tensor member
    std::vector<mlir::Type> world_members;
    world_members.reserve(1);
    world_members.push_back(tensor_type);
    mlir::Type structType = mlir::bm::WorldType::get(world_members);
    mlir::ArrayAttr dataAttr = builder.getArrayAttr(attribute);
    mlir::bm::WorldConstantOp struct_op = builder.create<mlir::bm::WorldConstantOp>(generated_loc(), structType, dataAttr);
    theModule.push_back(struct_op);

    // create a struct type that has a pointer to tensors. allocate two tensors, inject them into struct, and then delete both tensors


    // Verify the module after we have finished constructing it, this will check
    // the structural properties of the IR and invoke any specific verifiers we
    // have on the Toy operations.
    if (failed(mlir::verify(theModule))) {
        theModule.emitError("module verification error");
        return nullptr;
    }
    theModule->dump();
    return theModule;
}