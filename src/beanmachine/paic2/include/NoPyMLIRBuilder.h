//
// Created by Steffi Stumpos on 7/25/22.
//

#ifndef PAIC2_NOPYMLIRBUILDER_H
#define PAIC2_NOPYMLIRBUILDER_H
#include "mlir-c/IR.h"
#include "NoPyPaicAST.h"
#include <vector>
using Tensor = std::vector<float, std::allocator<float>>;

namespace nopy {
    class WorldSpec {
    public:
        WorldSpec():_world_size(0){ init_data = nullptr; }
        WorldSpec(WorldSpec const& copy):_print_name(copy.print_name()), _world_size(copy.world_size()), init_data(copy.init_data) {}
        void set_print_name(std::string name){ _print_name = name; }
        void set_world_name(std::string name) { _world_name = name; }
        void set_world_size(int size){
            if(size < 0){
                _world_size = 0;
            } else {
                _world_size = size;
            }
        }
        std::string print_name()const{return _print_name;}
        std::string world_name()const{return _world_name;}
        int world_size()const{return _world_size; }

        std::shared_ptr<std::vector<double>> init_data;
    private:
        std::string _print_name;
        std::string _world_name;
        int _world_size;
    };
    class NoPyMLIRBuilder {
    public:
        NoPyMLIRBuilder();
        void print_func_name(std::shared_ptr<nopy::PythonFunction> function);
        void infer(std::shared_ptr<nopy::PythonFunction> function, WorldSpec const& worldClassSpec, std::shared_ptr<std::vector<double>> init_nodes);
    };
}
#endif //PAIC2_NOPYMLIRBUILDER_H
