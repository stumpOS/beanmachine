//
// Created by Steffi Stumpos on 6/24/22.
//

#ifndef PAIC_IR_WORLDTYPEBUILDER_H
#define PAIC_IR_WORLDTYPEBUILDER_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "WorldClassSpec.h"

#include "llvm/ADT/ScopedHashTable.h"

namespace mlir {
    class MLIRContext;
    template <typename OpTy>
    class OwningOpRef;
    class ModuleOp;
} // namespace mlir

namespace paic_mlir {
    class WorldTypeBuilder {
    public:
        WorldTypeBuilder(mlir::MLIRContext &context);
        // the module will contain a type
        mlir::ModuleOp generate_op(paic_mlir::WorldClassSpec const& worldClassSpec);
    private:
        mlir::ModuleOp theModule;
        mlir::OpBuilder builder;
        llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symbolTable;
        llvm::ScopedHashTable<llvm::StringRef, mlir::func::FuncOp> functionSymbolTable;
    };
}
#endif //PAIC_IR_WORLDTYPEBUILDER_H
