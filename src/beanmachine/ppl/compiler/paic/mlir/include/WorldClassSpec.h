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

    class WorldClassSpec {
    public:
        WorldClassSpec(std::vector<LogProbQueryTypes> funcsToGenerate):_functions(funcsToGenerate){}
    private:
        std::vector<LogProbQueryTypes> _functions;
    };
}


#endif //PAIC_IR_WORLDCLASSSPEC_H
