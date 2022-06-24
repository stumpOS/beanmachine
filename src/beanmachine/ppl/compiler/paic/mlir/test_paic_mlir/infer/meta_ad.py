import abc

import torch
import typing
# TODO: AD and World should be apart of the same context because you cannot use PyTorchAD with a tracer
import beanmachine.ppl.utils.tensorops


class AD(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def jacobian_and_hessian(self, x:torch.Tensor, func:typing.Callable[[torch.Tensor],torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError()

class PyTorchAD(AD):
    def jacobian_and_hessian(self, x:torch.Tensor, func:typing.Callable[[torch.Tensor],torch.Tensor]) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        y = x.clone()
        y.requires_grad = True
        backprop_tree = func(y)
        return beanmachine.ppl.utils.tensorops.gradients(backprop_tree, y)

AD.register(PyTorchAD)