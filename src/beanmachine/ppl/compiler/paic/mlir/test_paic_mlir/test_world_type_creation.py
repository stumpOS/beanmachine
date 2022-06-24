import paic_mlir

def test_create_type():
    functions_to_generate = paic_mlir.LogProbQueryList()
    functions_to_generate.push_back(paic_mlir.LogProbQueryTypes.TARGET)
    mb = paic_mlir.MLIRBuilder()
    world_type_constructor = mb.create_world_constructor(paic_mlir.WorldClassSpec(functions_to_generate))

    print("compiles")

if __name__ == "__main__":
    test_create_type()