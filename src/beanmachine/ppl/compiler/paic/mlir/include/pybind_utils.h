//
// Created by Steffi Stumpos on 6/24/22.
//

#ifndef PAIC_IR_PYBIND_UTILS_H
#define PAIC_IR_PYBIND_UTILS_H
#include <pybind11/stl.h>

namespace py = pybind11;
template<typename T>
void bind_vector(pybind11::module &m, const char* name){
    py::class_<std::vector<T, std::allocator<T>>>(m, name)
            .def(py::init<>())
            .def("pop_back", &std::vector<T>::pop_back)
                    /* There are multiple versions of push_back(), etc. Select the right ones. */
            .def("push_back", (void(std::vector<T>::*)(const T &)) &std::vector<T>::push_back)
            .def("back", (T & (std::vector<T>::*) ()) & std::vector<T>::back)
            .def("__len__", [](const std::vector<T> &v) { return v.size(); })
            .def("__iter__",
                 [](std::vector<T> &v) { return py::make_iterator(v.begin(), v.end()); },
                 py::keep_alive<0, 1>());
}
#endif //PAIC_IR_PYBIND_UTILS_H
