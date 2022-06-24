//
// Created by Steffi Stumpos on 6/10/22.
//

#ifndef PAIC_IR_MLIRBUILDER_H
#define PAIC_IR_MLIRBUILDER_H

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include "mlir-c/IR.h"
#include "PaicAST.h"
#include "WorldClassSpec.h"

namespace paic_mlir {
    class MLIRBuilder {
    public:
        static void bind(pybind11::module &m);
        MLIRBuilder(pybind11::object contextObj);
        pybind11::float_ to_metal(std::shared_ptr<paic_mlir::PythonFunction> function, pybind11::float_ input);
        // The goal of this exercise is to see if we can create an instance of a type where the type
        pybind11::cpp_function create_constructor_for_type(paic_mlir::WorldClassSpec const& worldClassSpec);
    };
}

#endif //PAIC_IR_MLIRBUILDER_H
