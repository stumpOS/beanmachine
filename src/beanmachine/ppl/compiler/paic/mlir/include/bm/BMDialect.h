//
// Created by Steffi Stumpos on 6/24/22.
//

#ifndef PAIC_IR_BMDIALECT_H
#define PAIC_IR_BMDIALECT_H
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// This class defines the struct type. It represents a collection
// of function pointers and an array.
namespace mlir {
    namespace bm {
        struct WorldTypeStorage;
    }
}
#include "bm/bm_dialect.h.inc"
#define GET_OP_CLASSES
#include "bm/bm_ops.h.inc"
//===----------------------------------------------------------------------===//
// BM Types
//===----------------------------------------------------------------------===//
namespace mlir {
    namespace bm {
/// This class defines the PAIC struct type. It represents a collection of
/// element types. All derived types in MLIR must inherit from the CRTP class
/// 'Type::TypeBase'. It takes as template parameters the concrete type
/// (StructType), the base class to use (Type), and the storage class
/// (StructTypeStorage).
        class WorldType : public mlir::Type::TypeBase<WorldType, mlir::Type,
                WorldTypeStorage> {
        public:
            /// Inherit some necessary constructors from 'TypeBase'.
            using Base::Base;

            /// Create an instance of a `StructType` with the given element types. There
            /// *must* be atleast one element type.
            static WorldType get(llvm::ArrayRef<mlir::Type> elementTypes);

            /// Returns the element types of this struct type.
            llvm::ArrayRef<mlir::Type> getElementTypes();

            /// Returns the number of element type held by this struct.
            size_t getNumElementTypes() { return getElementTypes().size(); }
        };
    }
}

#endif //PAIC_IR_BMDIALECT_H
