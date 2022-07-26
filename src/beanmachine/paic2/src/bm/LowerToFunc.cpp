//
// Created by Steffi Stumpos on 7/20/22.
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
// MemRef Helper functions
static MemRefType convertTensorToMemRef(TensorType type) {
    assert(type.hasRank() && "expected only ranked shapes");
    return MemRefType::get(type.getShape(), type.getElementType());
}

static Value insertAllocAndDealloc(MemRefType type, Location loc,
                                   PatternRewriter &rewriter) {
    auto alloc = rewriter.create<memref::AllocOp>(loc, type);
    auto *parentBlock = alloc->getBlock();
    alloc->moveBefore(&parentBlock->front());
    auto dealloc = rewriter.create<memref::DeallocOp>(loc, alloc);
    dealloc->moveBefore(&parentBlock->back());
    return alloc;
}
//===----------------------------------------------------------------------===//
// BMToFunc RewritePatterns: Return operations
//===----------------------------------------------------------------------===//

struct ReturnOpLowering : public OpRewritePattern<bm::ReturnOp> {
    using OpRewritePattern<bm::ReturnOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(bm::ReturnOp op, PatternRewriter &rewriter) const final {
        if (op.hasOperand()) {
            rewriter.create<func::ReturnOp>(op->getLoc(), op->getResultTypes(), op->getOperands());
            rewriter.eraseOp(op);
            return success();
        }

        // We lower "toy.return" directly to "func.return".
        rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
        return success();
    }
};
//===----------------------------------------------------------------------===//
// BMToFunc RewritePatterns: Print operations
//===----------------------------------------------------------------------===//

struct PrintOpLowering : public OpConversionPattern<bm::PrintWorldOp> {
    using OpConversionPattern<bm::PrintWorldOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(bm::PrintWorldOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const final {
        // We don't lower "bm.print_world" in this pass, but we need to update its
        // operands.
        rewriter.updateRootInPlace(op,[&] { op->setOperands(adaptor.getOperands()); });
        return success();
    }
};

//===----------------------------------------------------------------------===//
// BMToAffine RewritePatterns: Constant operations
//===----------------------------------------------------------------------===//

struct WorldConstantOpLowering : public OpRewritePattern<mlir::bm::WorldConstantOp> {
    using OpRewritePattern<mlir::bm::WorldConstantOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(mlir::bm::WorldConstantOp op,
                                  PatternRewriter &rewriter) const final {
        ArrayAttr arrayAttr = op.getValue();
        if(arrayAttr.size() != 1){
            return failure();
        }
        DenseElementsAttr constantValue = arrayAttr.begin()->dyn_cast_or_null<DenseElementsAttr>();
        Location loc = op.getLoc();

        // When lowering the constant operation, we allocate and assign the constant
        // values to a corresponding memref allocation.
        auto worldType = op.getType().cast<bm::WorldType>();
        auto tensorType = worldType.getElementTypes().front();
        auto memRefType = convertTensorToMemRef(tensorType.cast<TensorType>());
        auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

        // We will be generating constant indices up-to the largest dimension.
        // Create these constants up-front to avoid large amounts of redundant
        // operations.
        auto valueShape = memRefType.getShape();
        SmallVector<Value, 8> constantIndices;

        if (!valueShape.empty()) {
            for (auto i : llvm::seq<int64_t>(
                    0, *std::max_element(valueShape.begin(), valueShape.end())))
                constantIndices.push_back(
                        rewriter.create<arith::ConstantIndexOp>(loc, i));
        } else {
            // This is the case of a tensor of rank 0.
            constantIndices.push_back(
                    rewriter.create<arith::ConstantIndexOp>(loc, 0));
        }

        // The constant operation represents a multi-dimensional constant, so we
        // will need to generate a store for each of the elements. The following
        // functor recursively walks the dimensions of the constant shape,
        // generating a store when the recursion hits the base case.
        SmallVector<Value, 2> indices;
        auto valueIt = constantValue.value_begin<FloatAttr>();
        std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
            // The last dimension is the base case of the recursion, at this point
            // we store the element at the given index.
            if (dimension == valueShape.size()) {
                rewriter.create<AffineStoreOp>(
                        loc, rewriter.create<arith::ConstantOp>(loc, *valueIt++), alloc,
                        llvm::makeArrayRef(indices));
                return;
            }

            // Otherwise, iterate over the current dimension and add the indices to
            // the list.
            for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
                indices.push_back(constantIndices[i]);
                storeElements(dimension + 1);
                indices.pop_back();
            }
        };

        // Start the element storing recursion from the first dimension.
        storeElements(/*dimension=*/0);

        // Replace this operation with the generated alloc.
        rewriter.replaceOp(op, alloc);
        return success();
    }
};
//===----------------------------------------------------------------------===//
// BMToFunc RewritePatterns: Func operations
//===----------------------------------------------------------------------===//

struct FuncOpLowering : public OpConversionPattern<bm::FuncOp> {
    using OpConversionPattern<bm::FuncOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(bm::FuncOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const final {
        llvm::SmallVector<mlir::Type> types;
        for(mlir::Type type_input : op.getFunctionType().getInputs()){
            auto world_ptr = type_input.dyn_cast_or_null<bm::WorldType>();
            if(world_ptr != nullptr){
                mlir::ShapedType rankedTensorType = world_ptr.getElementTypes().front();
                mlir::ShapedType rankedMemRefType = mlir::MemRefType::get(rankedTensorType.getShape(), rankedTensorType.getElementType());
                types.push_back(rankedMemRefType);
            } else {
                types.push_back(type_input);
            }
        }

        // TODO: needs to be recursive. Also use TypeConverter?
        for(auto arg_ptr = op.front().args_begin(); arg_ptr != op.front().args_end(); arg_ptr++){
            mlir::BlockArgument blockArgument = *arg_ptr;
            auto world_ptr = blockArgument.getType().dyn_cast_or_null<bm::WorldType>();
            if(world_ptr != nullptr) {
                mlir::ShapedType rankedTensorType = world_ptr.getElementTypes().front();
                mlir::ShapedType rankedMemRefType = mlir::MemRefType::get(rankedTensorType.getShape(), rankedTensorType.getElementType());
                blockArgument.setType(rankedMemRefType);
            }
        }
        mlir::TypeRange typeRange(types);
        mlir::FunctionType new_function_type = rewriter.getFunctionType(typeRange, op.getFunctionType().getResults());

        // Create a new function with an updated signature.
        auto newFuncOp = rewriter.create<mlir::func::FuncOp>(op.getLoc(), op.getName().str(), new_function_type);
        newFuncOp->setAttrs(op->getAttrs());
        newFuncOp.setType(new_function_type);
        rewriter.inlineRegionBefore(op.getBody(), newFuncOp.getBody(),newFuncOp.end());
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
                             [](Type type) {
                                 bool is_world = type.isa<bm::WorldType>();
                                 return is_world;
                             });
    });

    // Now that the conversion target has been defined, we just need to provide
    // the set of patterns that will lower the Toy operations.
    RewritePatternSet patterns(&getContext());
    patterns.add<FuncOpLowering, PrintOpLowering, ReturnOpLowering, WorldConstantOpLowering>(&getContext());

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