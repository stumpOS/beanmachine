//
// Created by Steffi Stumpos on 6/28/22.
//

#ifndef PAIC_IR_PASSES_H
#define PAIC_IR_PASSES_H
#include <memory>

namespace mlir {
    class Pass;

    namespace bm {
        std::unique_ptr<mlir::Pass> createLowerToFuncPass();
        std::unique_ptr<mlir::Pass> createLowerToLLVMPass();
    }
}
#endif //PAIC_IR_PASSES_H
