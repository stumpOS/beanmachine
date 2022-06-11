//
// Created by Steffi Stumpos on 6/10/22.
//

#ifndef PAIC_IR_MLIRBUILDER_H
#define PAIC_IR_MLIRBUILDER_H

#include <pybind11/pybind11.h>
#include "mlir-c/IR.h"

namespace paic_mlir {
    class MLIRBuilder {
    public:
        static void bind(pybind11::module &m);
        MLIRBuilder(pybind11::object contextObj);
    };
}

#endif //PAIC_IR_MLIRBUILDER_H
