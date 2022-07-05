//
// Created by Steffi Stumpos on 6/24/22.
//

#ifndef PAIC_IR_WORLDCLASSSPEC_H
#define PAIC_IR_WORLDCLASSSPEC_H
#include <vector>

namespace paic_mlir {
    // TODO: more general graph specification
    enum LogProbQueryTypes {
        TARGET_AND_CHILDREN,
        ALL_LATENT_VARIABLES,
        TARGET
    };

    // The compiler knows that the world has certain members and the layout of those members, it just doesn't know what those
    // members are called. This is the configuration for the world fields.
    class WorldClassSpec {
    public:
        WorldClassSpec(std::vector<LogProbQueryTypes> funcsToGenerate):_functions(funcsToGenerate){}
        unsigned query_function_count() const { return _functions.size(); }
        void set_print_name(std::string name){ _print_name = name; }
        void set_world_name(std::string name) { _world_name = name; }
        std::string print_name()const{return _print_name;}
        std::string world_name()const{return _world_name;}
    private:
        std::vector<LogProbQueryTypes> _functions;
        std::string _print_name;
        std::string _world_name;
    };
}


#endif //PAIC_IR_WORLDCLASSSPEC_H
