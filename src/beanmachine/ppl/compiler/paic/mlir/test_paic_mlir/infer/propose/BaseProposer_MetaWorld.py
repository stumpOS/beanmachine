import abc
import torch
from typing import Tuple

from beanmachine.ppl.compiler.paic.mlir.test_paic_mlir.infer.meta_world import MetaWorld


class BaseProposer_MetaWorld(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def propose(self, world: MetaWorld) -> Tuple[MetaWorld, torch.Tensor]:
        raise NotImplementedError

    def do_adaptation(self, world, accept_log_prob, *args, **kwargs) -> None:
        ...

    def finish_adaptation(self) -> None:
        ...