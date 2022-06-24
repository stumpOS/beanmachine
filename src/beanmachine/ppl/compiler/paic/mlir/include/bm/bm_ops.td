#ifndef BM_OPS
#define BM_OPS

include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Provide a definition of the 'bm' dialect in the ODS framework so that we
// can define our operations.
def BM_Dialect : Dialect {
  let name = "bm";
  let cppNamespace = "::mlir::bm";
  let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;
  let hasConstantMaterializer = 1;
  let useDefaultTypePrinterParser = 1;
}
// Base class for bm dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class BM_Op<string mnemonic, list<Trait> traits = []> :
Op<BM_Dialect, mnemonic, traits>;

// Provide a definition for the BM StructType for use in ODS. This allows for
// using StructType in a similar way to Tensor or MemRef. We use `DialectType`
// to demarcate the StructType as belonging to the BM dialect.
def BM_StructType :
DialectType<BM_Dialect, CPred<"$_self.isa<StructType>()">,
        "BM struct type">;

// this allows us to specify for example that a call operations has an out type of either BM_StructType or F32.
// in reality it can also return a pointer but we are not there yet.
def BM_Type : AnyTypeOf<[F32, BM_StructType]>;
//===----------------------------------------------------------------------===//
// StructAccessOp
//===----------------------------------------------------------------------===//

def StructAccessOp : BM_Op<"struct_access", [NoSideEffect]> {
    let summary = "struct access";
    let description = [{
        Access the Nth element of a value returning a struct type.
    }];

    let arguments = (ins BM_StructType:$input, I64Attr:$index);
    let results = (outs BM_Type:$output);

    let assemblyFormat = [{
        $input `[` $index `]` attr-dict `:` type($input) `->` type($output)
    }];

    // Allow building a StructAccessOp with just a struct value and an index.
    let builders = [
        OpBuilder<(ins "Value":$input, "size_t":$index)>
    ];

    // Indicate that additional verification for this operation is necessary.
    let hasVerifier = 1;

    // Set the folder bit so that we can fold constant accesses.
    let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// StructConstantOp
//===----------------------------------------------------------------------===//

def StructConstantOp : BM_Op<"struct_constant", [ConstantLike, NoSideEffect]> {
    let summary = "struct constant";
    let description = [{
        Constant operation turns a literal struct value into an SSA value. The data
        is attached to the operation as an attribute. The struct constant is encoded
        as an array of other constant values. For example:

        ```mlir
            %0 = bm.struct_constant [
            dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
            ] : !bm.struct<tensor<*xf64>>
        ```
    }];

    let arguments = (ins ArrayAttr:$value);
    let results = (outs BM_StructType:$output);

    let assemblyFormat = "$value attr-dict `:` type($output)";

    // Indicate that additional verification for this operation is necessary.
    let hasVerifier = 1;
    let hasFolder = 1;
}
#endif // BM_OPS