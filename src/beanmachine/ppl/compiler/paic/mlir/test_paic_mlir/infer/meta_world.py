from __future__ import annotations

import abc
import dataclasses
import typing
from typing import (
    Collection,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    TypeVar,
)

import torch
from torch.distributions import Distribution

import beanmachine.ppl.legacy.inference.proposer.newtonian_monte_carlo_utils
import beanmachine.ppl.utils.tensorops
import beanmachine.ppl.world
from beanmachine.ppl import RVIdentifier
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.world import init_to_uniform
from beanmachine.ppl.world.initialize_fn import init_from_prior, InitializeFn
from beanmachine.ppl.legacy.inference.proposer.newtonian_monte_carlo_utils import hessian_of_log_prob

T = TypeVar("T", bound="World")
from abc import ABCMeta, abstractmethod
from typing import Tuple

import torch


class RVMeta:
    def __init__(self, constraint: torch.distributions.constraints.Constraint, type: typing.ClassVar):
        self._support = constraint
        self._type = type

    def support(self) -> torch.distributions.constraints.Constraint:
        return self._support

    def distribution(self) -> typing.ClassVar:
        return self._type


# a meta world has two implementations:
# (1) a tracer
# (2) Python implementation
class MetaWorld(metaclass=ABCMeta):
    def __init__(self, queries: Iterable[RVIdentifier],
                 observations: Dict[RVIdentifier, torch.Tensor]):
        self._queries = queries
        self._observations = observations

    @abstractmethod
    def queries(self) -> Iterable[RVIdentifier]:
        return self._queries

    @abstractmethod
    def observations(self) -> Dict[RVIdentifier, torch.Tensor]:
        return self._observations

    @abstractmethod
    def value_of(self, z: RVIdentifier) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def latent_variable_count(self) -> int:
        return list(self._queries).__len__()

    @abstractmethod
    def log_prob(self, of: Optional[Collection[RVIdentifier]] = None) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def rv_metadata(self, RVIdentifier) -> torch.distributions.constraints.Constraint:
        raise NotImplementedError()

    @abstractmethod
    def replace(self, rv: RVIdentifier, value: torch.Tensor) -> MetaWorld:
        raise NotImplementedError()

    @abstractmethod
    def hessian_of_log_prob_of_children_given_target(self, target: RVIdentifier, value: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        raise NotImplementedError()


def copy_from_world(python_world: beanmachine.ppl.world.World) -> RealWorld:
    rw = RealWorld([], {})
    rw.python_world = python_world
    return rw


class RealWorld(MetaWorld):
    def __init__(self, queries: Iterable[RVIdentifier], observations: Dict[RVIdentifier, torch.Tensor]):
        super().__init__(queries, observations)
        self.python_world = beanmachine.ppl.world.World.initialize_world(queries, observations)

    def queries(self) -> Iterable[RVIdentifier]:
        return self.python_world.latent_nodes

    def observations(self) -> Dict[RVIdentifier, torch.Tensor]:
        return self.python_world.observations

    def value_of(self, z: RVIdentifier) -> torch.Tensor:
        return self.python_world[z]

    def latent_variable_count(self) -> int:
        return list(self._queries).__len__()

    def log_prob(self, of: Optional[Collection[RVIdentifier]] = None) -> torch.Tensor:
        return self.python_world.log_prob(of)

    def rv_metadata(self, rv: RVIdentifier) -> torch.distributions.constraints.Constraint:
        variable = self.python_world.get_variable(rv)
        return RVMeta(variable.distribution.support, type(variable.distribution))

    def hessian_of_log_prob_of_children_given_target(self, target: RVIdentifier, value: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        return hessian_of_log_prob(self.python_world, target, value, beanmachine.ppl.utils.tensorops.gradients)

    def replace(self, rv: RVIdentifier, value: torch.Tensor) -> MetaWorld:
        rw = copy_from_world(python_world=self.python_world.replace({rv: value}))
        return rw


MetaWorld.register(RealWorld)
