//
// Created by Steffi Stumpos on 6/10/22.
//
#include "MLIRBuilder.h"

namespace py = pybind11;

PYBIND11_MODULE(paic_mlir, m) {
    m.doc() = "MVP for pybind module";
    paic_mlir::MLIRBuilder::bind(m);
}

void paic_mlir::MLIRBuilder::bind(py::module &m) {
    py::class_<MLIRBuilder>(m, "MLIRBuilder")
            .def(py::init<py::object>(), py::arg("context") = py::none());
}

paic_mlir::MLIRBuilder::MLIRBuilder(pybind11::object contextObj) {

}