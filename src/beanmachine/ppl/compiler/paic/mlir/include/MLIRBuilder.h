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
#include "AbstractWorld.h"
#include <iostream>

namespace paic_mlir {
    struct InferenceFunctions {
        InferenceFunctions(){}
        InferenceFunctions(pybind11::object contextObj);
        const pybind11::object create_world(std::shared_ptr<std::vector<float>> vars) const {
            return _create_world(vars);
        }
        void inference_function(std::shared_ptr<paic_mlir::AbstractWorld> res) const {
            _inference_function(res);
        }

        // accepts the initial values of the variables and returns a world instance
        pybind11::cpp_function _create_world;
        // accepts an instance of a world and prints all values in the world
        pybind11::cpp_function _inference_function;
    };
    class MLIRBuilder {
    public:
        static void bind(pybind11::module &m);
        MLIRBuilder(pybind11::object contextObj);
        pybind11::float_ to_metal(std::shared_ptr<paic_mlir::PythonFunction> function, pybind11::float_ input);
        // The Python function will contain calls on a "World" type. In order to generate the MLIR we have to know the layout of that type
        std::shared_ptr<InferenceFunctions> create_inference_functions(std::shared_ptr<paic_mlir::PythonFunction> function, paic_mlir::WorldClassSpec const& worldClassSpec);
    };
}

#endif //PAIC_IR_MLIRBUILDER_H
