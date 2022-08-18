//
// Created by Steffi Stumpos on 6/3/22.
//

#ifndef PPL_TOYDIALECT_H
#define PPL_TOYDIALECT_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

/// Include the auto-generated header file containing the declaration of the toy
/// dialect.
#include "dialect.h.inc"
#include "ShapeInferenceInterface.h"

/// Include the auto-generated header file containing the declarations of the
/// toy operations.
#define GET_OP_CLASSES
#include "ops.h.inc"
#endif //PPL_TOYDIALECT_H
