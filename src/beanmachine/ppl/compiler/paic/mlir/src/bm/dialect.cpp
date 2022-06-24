//
// Created by Steffi Stumpos on 6/24/22.
//
#include "bm/BMDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::bm;

#include "bm/bm_dialect.cpp.inc"

#define GET_OP_CLASSES
#include "bm/bm_ops.cpp.inc"

void BMDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "bm/bm_ops.cpp.inc"
    >();
    addTypes<StructType>();
}

//===----------------------------------------------------------------------===//
// BM Types
//===----------------------------------------------------------------------===//

namespace mlir {
    namespace bm {
        struct StructTypeStorage : public mlir::TypeStorage {
            /// The `KeyTy` is a required type that provides an interface for the storage
            /// instance. This type will be used when uniquing an instance of the type
            /// storage. For our struct type, we will unique each instance structurally on
            /// the elements that it contains.
            using KeyTy = llvm::ArrayRef<mlir::Type>;

            /// A constructor for the type storage instance.
            StructTypeStorage(llvm::ArrayRef<mlir::Type> elementTypes)
                    : elementTypes(elementTypes) {}

            /// Define the comparison function for the key type with the current storage
            /// instance. This is used when constructing a new instance to ensure that we
            /// haven't already uniqued an instance of the given key.
            bool operator==(const KeyTy &key) const { return key == elementTypes; }

            /// Define a hash function for the key type. This is used when uniquing
            /// instances of the storage, see the `StructType::get` method.
            /// Note: This method isn't necessary as both llvm::ArrayRef and mlir::Type
            /// have hash functions available, so we could just omit this entirely.
            static llvm::hash_code hashKey(const KeyTy &key) {
                return llvm::hash_value(key);
            }

            /// Define a construction function for the key type from a set of parameters.
            /// These parameters will be provided when constructing the storage instance
            /// itself.
            /// Note: This method isn't necessary because KeyTy can be directly
            /// constructed with the given parameters.
            static KeyTy getKey(llvm::ArrayRef<mlir::Type> elementTypes) {
                return KeyTy(elementTypes);
            }

            /// Define a construction method for creating a new instance of this storage.
            /// This method takes an instance of a storage allocator, and an instance of a
            /// `KeyTy`. The given allocator must be used for *all* necessary dynamic
            /// allocations used to create the type storage and its internal.
            static StructTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                                const KeyTy &key) {
                // Copy the elements from the provided `KeyTy` into the allocator.
                llvm::ArrayRef<mlir::Type> elementTypes = allocator.copyInto(key);

                // Allocate the storage instance and construct it.
                return new(allocator.allocate<StructTypeStorage>())
                        StructTypeStorage(elementTypes);
            }

            /// The following field contains the element types of the struct.
            llvm::ArrayRef<mlir::Type> elementTypes;
        };
    }
}

/// Create an instance of a `StructType` with the given element types. There
/// *must* be at least one element type.
StructType StructType::get(llvm::ArrayRef<mlir::Type> elementTypes) {
    assert(!elementTypes.empty() && "expected at least 1 element type");

    // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
    // of this type. The first parameter is the context to unique in. The
    // parameters after the context are forwarded to the storage instance.
    mlir::MLIRContext *ctx = elementTypes.front().getContext();
    return Base::get(ctx, elementTypes);
}

/// Returns the element types of this struct type.
llvm::ArrayRef<mlir::Type> StructType::getElementTypes() {
    // 'getImpl' returns a pointer to the internal storage instance.
    return getImpl()->elementTypes;
}

/// Parse an instance of a type registered to the toy dialect.
mlir::Type BMDialect::parseType(mlir::DialectAsmParser &parser) const {
    // Parse a struct type in the following form:
    //   struct-type ::= `struct` `<` type (`,` type)* `>`

    // NOTE: All MLIR parser function return a ParseResult. This is a
    // specialization of LogicalResult that auto-converts to a `true` boolean
    // value on failure to allow for chaining, but may be used with explicit
    // `mlir::failed/mlir::succeeded` as desired.

    // Parse: `struct` `<`
    if (parser.parseKeyword("struct") || parser.parseLess())
        return Type();

    // Parse the element types of the struct.
    SmallVector<mlir::Type, 1> elementTypes;
    do {
        // Parse the current element type.
        SMLoc typeLoc = parser.getCurrentLocation();
        mlir::Type elementType;
        if (parser.parseType(elementType))
            return nullptr;

        // Check that the type is either a TensorType or another StructType.
        if (!elementType.isa<mlir::TensorType, StructType>()) {
            parser.emitError(typeLoc, "element type for a struct must either "
                                      "be a TensorType or a StructType, got: ")
                    << elementType;
            return Type();
        }
        elementTypes.push_back(elementType);

        // Parse the optional: `,`
    } while (succeeded(parser.parseOptionalComma()));

    // Parse: `>`
    if (parser.parseGreater())
        return Type();
    return StructType::get(elementTypes);
}

/// Print an instance of a type registered to the toy dialect.
void BMDialect::printType(mlir::Type type,
                           mlir::DialectAsmPrinter &printer) const {
    // Currently the only toy type is a struct type.
    StructType structType = type.cast<StructType>();

    // Print the struct type according to the parser format.
    printer << "struct<";
    llvm::interleaveComma(structType.getElementTypes(), printer);
    printer << '>';
}

mlir::Operation *BMDialect::materializeConstant(mlir::OpBuilder &builder,
                                                 mlir::Attribute value,
                                                 mlir::Type type,
                                                 mlir::Location loc) {
    if (type.isa<StructType>())
        return builder.create<StructConstantOp>(loc, type,
                                                value.cast<mlir::ArrayAttr>());
    assert(false);
}
void StructAccessOp::build(mlir::OpBuilder &b, mlir::OperationState &state,
                           mlir::Value input, size_t index) {
    // Extract the result type from the input type.
    StructType structTy = input.getType().cast<StructType>();
    assert(index < structTy.getNumElementTypes());
    mlir::Type resultType = structTy.getElementTypes()[index];

    // Call into the auto-generated build method.
    build(b, state, resultType, input, b.getI64IntegerAttr(index));
}

mlir::LogicalResult StructAccessOp::verify() {
    StructType structTy = getInput().getType().cast<StructType>();
    size_t indexValue = getIndex();
    if (indexValue >= structTy.getNumElementTypes())
        return emitOpError()
                << "index should be within the range of the input struct type";
    mlir::Type resultType = getResult().getType();
    if (resultType != structTy.getElementTypes()[indexValue])
        return emitOpError() << "must have the same result type as the struct "
                                "element referred to by the index";
    return mlir::success();
}
/// Verify that the given attribute value is valid for the given type.
static mlir::LogicalResult verifyConstantForType(mlir::Type type,
                                                 mlir::Attribute opaqueValue,
                                                 mlir::Operation *op) {
    if (type.isa<mlir::TensorType>()) {
        // Check that the value is an elements attribute.
        auto attrValue = opaqueValue.dyn_cast<mlir::DenseFPElementsAttr>();
        if (!attrValue)
            return op->emitError("constant of TensorType must be initialized by "
                                 "a DenseFPElementsAttr, got ")
                    << opaqueValue;

        // If the return type of the constant is not an unranked tensor, the shape
        // must match the shape of the attribute holding the data.
        auto resultType = type.dyn_cast<mlir::RankedTensorType>();
        if (!resultType)
            return success();

        // Check that the rank of the attribute type matches the rank of the
        // constant result type.
        auto attrType = attrValue.getType().cast<mlir::TensorType>();
        if (attrType.getRank() != resultType.getRank()) {
            return op->emitOpError("return type must match the one of the attached "
                                   "value attribute: ")
                    << attrType.getRank() << " != " << resultType.getRank();
        }

        // Check that each of the dimensions match between the two types.
        for (int dim = 0, dimE = attrType.getRank(); dim < dimE; ++dim) {
            if (attrType.getShape()[dim] != resultType.getShape()[dim]) {
                return op->emitOpError(
                        "return type shape mismatches its attribute at dimension ")
                        << dim << ": " << attrType.getShape()[dim]
                        << " != " << resultType.getShape()[dim];
            }
        }
        return mlir::success();
    }
    auto resultType = type.cast<StructType>();
    llvm::ArrayRef<mlir::Type> resultElementTypes = resultType.getElementTypes();

    // Verify that the initializer is an Array.
    auto attrValue = opaqueValue.dyn_cast<ArrayAttr>();
    if (!attrValue || attrValue.getValue().size() != resultElementTypes.size())
        return op->emitError("constant of StructType must be initialized by an "
                             "ArrayAttr with the same number of elements, got ")
                << opaqueValue;

    // Check that each of the elements are valid.
    llvm::ArrayRef<mlir::Attribute> attrElementValues = attrValue.getValue();
    for (const auto it : llvm::zip(resultElementTypes, attrElementValues))
        if (failed(verifyConstantForType(std::get<0>(it), std::get<1>(it), op)))
            return mlir::failure();
    return mlir::success();
}

mlir::LogicalResult StructConstantOp::verify() {
    return verifyConstantForType(getResult().getType(), getValue(), *this);
}

/// Fold struct constants.
OpFoldResult StructConstantOp::fold(ArrayRef<Attribute> operands) {
    return getValue();
}

/// Fold simple struct access operations that access into a constant.
OpFoldResult StructAccessOp::fold(ArrayRef<Attribute> operands) {
    auto structAttr = operands.front().dyn_cast_or_null<mlir::ArrayAttr>();
    if (!structAttr)
        return nullptr;

    size_t elementIndex = getIndex();
    return structAttr[elementIndex];
}