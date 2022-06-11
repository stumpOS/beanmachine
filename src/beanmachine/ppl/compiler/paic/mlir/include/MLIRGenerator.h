//
// Created by Steffi Stumpos on 6/3/22.
//

#ifndef PPL_MLIRGENERATOR_H
#define PPL_MLIRGENERATOR_H
#include <memory>

namespace mlir {
    class MLIRContext;
    template <typename OpTy>
    class OwningOpRef;
    class ModuleOp;
} // namespace mlir

namespace demo {
    class ModuleAST;

/// Emit IR for the given Toy moduleAST, returns a newly created MLIR module
/// or nullptr on failure.
    mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context,
                                              ModuleAST &moduleAST);
}

#endif //PPL_MLIRGENERATOR_H
