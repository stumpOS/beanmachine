//
// Created by Steffi Stumpos on 6/24/22.
//
#include "WorldTypeBuilder.h"
#include "mlir/IR/Verifier.h"

paic_mlir::WorldTypeBuilder::WorldTypeBuilder(mlir::MLIRContext &context): builder(&context) {

}

mlir::ModuleOp paic_mlir::WorldTypeBuilder::generate_op(const paic_mlir::WorldClassSpec &worldClassSpec) {
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());


    // Verify the module after we have finished constructing it, this will check
    // the structural properties of the IR and invoke any specific verifiers we
    // have on the Toy operations.
    if (failed(mlir::verify(theModule))) {
        theModule.emitError("module verification error");
        return nullptr;
    }
    return theModule;
}