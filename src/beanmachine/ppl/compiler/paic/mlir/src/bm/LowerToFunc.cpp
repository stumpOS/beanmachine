//
// Created by Steffi Stumpos on 6/29/22.
//
#include "mlir/IR/BuiltinDialect.h"
#include "bm/BMDialect.h"
#include "bm/passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"
#include <iostream>

using namespace mlir;
//===----------------------------------------------------------------------===//
// BMToFunc RewritePatterns: Return operations
//===----------------------------------------------------------------------===//

struct ReturnOpLowering : public OpRewritePattern<bm::ReturnOp> {
    using OpRewritePattern<bm::ReturnOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(bm::ReturnOp op, PatternRewriter &rewriter) const final {
        // During this lowering, we expect that all function calls have been
        // inlined.
        if (op.hasOperand())
            return failure();

        // We lower "toy.return" directly to "func.return".
        rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
        return success();
    }
};
//===----------------------------------------------------------------------===//
// BMToFunc RewritePatterns: Func operations
//===----------------------------------------------------------------------===//

struct FuncOpLowering : public OpConversionPattern<bm::FuncOp> {
    using OpConversionPattern<bm::FuncOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(bm::FuncOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const final {
        bool accepts_world_type = op.getFunctionType().getInput(0).getTypeID() == bm::WorldType().getTypeID();
        // we do not yet map return types
        if (op.getFunctionType().getNumResults() > 0 || !accepts_world_type || op.getFunctionType().getNumResults() > 1) {
            return rewriter.notifyMatchFailure(op, [](Diagnostic &diag) {
                diag << "expected a bm function to accept only a world and return nothing";
            });
        }
        // currently we expect a bm function to accept a world and we expect a world to be just
        // be a wrapper around a 1D Tensor. So let's map that to an unranked MemRef type
        mlir::Attribute memSpace = mlir::IntegerAttr::get(mlir::IntegerType::get(rewriter.getContext(), 32), 7);
        mlir::ShapedType unrankedTensorType = mlir::UnrankedMemRefType::get(rewriter.getF32Type(), memSpace);
        mlir::FunctionType new_function_type = rewriter.getFunctionType({unrankedTensorType}, {});

        // Create a new function with an updated signature.
        auto newFuncOp = rewriter.create<mlir::func::FuncOp>(op.getLoc(), op.getName().str(), new_function_type);
        newFuncOp->dump();
        newFuncOp->setAttrs(op->getAttrs());
        newFuncOp.setType(new_function_type);
        op.front().eraseArgument(0);
        op.front().addArgument(unrankedTensorType, op.getLoc());
        rewriter.inlineRegionBefore(op.getBody(), newFuncOp.getBody(),newFuncOp.end());


        // TODO: use the type converter
        /*
        TypeConverter typeConverter;
        llvm::SmallVector<Type> sv;
        sv.push_back(unrankedTensorType);
        typeConverter.convertTypes(bm::WorldType::get({unrankedTensorType}), sv);
        rewriter.convertRegionTypes(op.getCallableRegion(), typeConverter);
        op.dump();
         */

        rewriter.eraseOp(op);

        return success();
    }
};

//===----------------------------------------------------------------------===//
// BMToFuncLoweringPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to affine loops of the toy operations that are
/// computationally intensive (like matmul for example...) while keeping the
/// rest of the code in the Toy dialect.
namespace {
    struct BMToFuncLoweringPass
            : public PassWrapper<BMToFuncLoweringPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BMToFuncLoweringPass)

    void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, memref::MemRefDialect>();
}
void runOnOperation() final;
};
} // namespace

void BMToFuncLoweringPass::runOnOperation() {
    ConversionTarget target(getContext());
    target.addLegalDialect<AffineDialect, BuiltinDialect, arith::ArithmeticDialect,func::FuncDialect, memref::MemRefDialect>();
    target.addIllegalDialect<bm::BMDialect>();
    // PrintWorld will be handled in a separate lowering but we need to lower its operands here
    target.addDynamicallyLegalOp<bm::PrintWorldOp>([](bm::PrintWorldOp op) {
        return llvm::none_of(op->getOperandTypes(),
                             [](Type type) { return type.isa<bm::WorldType>(); });
    });

    // Now that the conversion target has been defined, we just need to provide
    // the set of patterns that will lower the Toy operations.
    RewritePatternSet patterns(&getContext());
    patterns.add<FuncOpLowering, ReturnOpLowering>(&getContext());

    // With the target and rewrite patterns defined, we can now attempt the
    // conversion. The conversion will signal failure if any of our `illegal`
    // operations were not converted successfully.
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
        signalPassFailure();
}



/// Create a pass for lowering operations in the `Affine` and `Std` dialects,
/// for a subset of the Toy IR (e.g. matmul).
std::unique_ptr<Pass> mlir::bm::createLowerToFuncPass() {
    return std::make_unique<BMToFuncLoweringPass>();
}