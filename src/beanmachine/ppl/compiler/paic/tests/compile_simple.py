import unittest
from beanmachine.paic_mlir import MLIRBuilder

class CompileToMLIR(unittest.TestCase):
    def test_mlir_builder_ref(self) -> None:
        mb = MLIRBuilder()
        self.assertEqual(1, 0)