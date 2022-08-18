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


class MetaWorld(metaclass=ABCMeta):
    @abstractmethod
    def queries(self) -> Iterable[RVIdentifier]:
        raise NotImplementedError()

    @abstractmethod
    def observations(self) -> Dict[RVIdentifier, torch.Tensor]:
        raise NotImplementedError()

    @abstractmethod
    def value_of(self, z: RVIdentifier) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def latent_variable_count(self) -> int:
        raise NotImplementedError()

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
    def children_of(self, rv: RVIdentifier) -> typing.Set[RVIdentifier]:
        raise NotImplementedError()

    @abstractmethod
    def parents_of(self, rv: RVIdentifier) -> typing.Set[RVIdentifier]:
        raise NotImplementedError()

    @abstractmethod
    def print(self):
        raise NotImplementedError()

def copy_from_world(python_world: beanmachine.ppl.world.World) -> RealWorld:
    rw = RealWorld([], {})
    rw.python_world = python_world
    return rw


class RealWorld(MetaWorld):
    def __init__(self, queries: Iterable[RVIdentifier], observations: Dict[RVIdentifier, torch.Tensor]):
        self.python_world = beanmachine.ppl.world.World.initialize_world(queries, observations)

    def queries(self) -> Iterable[RVIdentifier]:
        return self.python_world.latent_nodes

    def observations(self) -> Dict[RVIdentifier, torch.Tensor]:
        return self.python_world.observations

    def value_of(self, z: RVIdentifier) -> torch.Tensor:
        return self.python_world[z]

    def latent_variable_count(self) -> int:
        return len(self.python_world.latent_nodes)

    def log_prob(self, of: Optional[Collection[RVIdentifier]] = None) -> torch.Tensor:
        return self.python_world.log_prob(of)

    def rv_metadata(self, rv: RVIdentifier) -> torch.distributions.constraints.Constraint:
        variable = self.python_world.get_variable(rv)
        return RVMeta(variable.distribution.support, type(variable.distribution))

    def children_of(self, rv: RVIdentifier) -> typing.Set[RVIdentifier]:
        return self.python_world.get_variable(rv).children

    def parents_of(self, rv: RVIdentifier) -> typing.Set[RVIdentifier]:
        return self.python_world.get_variable(rv).parents

    def replace(self, rv: RVIdentifier, value: torch.Tensor) -> MetaWorld:
        rw = copy_from_world(python_world=self.python_world.replace({rv: value}))
        return rw

    def print(self):
        print(str(self.python_world))

class TraceWorld(MetaWorld):
    def __init__(self, queries: Iterable[RVIdentifier], observations: Dict[RVIdentifier, torch.Tensor]):
        self.queries()
        self.statements = []

    def queries(self) -> Iterable[RVIdentifier]:
        raise NotImplementedError()

    def observations(self) -> Dict[RVIdentifier, torch.Tensor]:
        raise NotImplementedError()

    def value_of(self, z: RVIdentifier) -> torch.Tensor:
        raise NotImplementedError()

    def latent_variable_count(self) -> int:
        raise NotImplementedError()

    def log_prob(self, of: Optional[Collection[RVIdentifier]] = None) -> torch.Tensor:
        raise NotImplementedError()

    def rv_metadata(self, rv: RVIdentifier) -> torch.distributions.constraints.Constraint:
        raise NotImplementedError()

    def children_of(self, rv: RVIdentifier) -> typing.Set[RVIdentifier]:
        raise NotImplementedError()

    def parents_of(self, rv: RVIdentifier) -> typing.Set[RVIdentifier]:
        raise NotImplementedError()

    def replace(self, rv: RVIdentifier, value: torch.Tensor) -> MetaWorld:
        raise NotImplementedError()

    def print(self):
        raise NotImplementedError()

MetaWorld.register(RealWorld)
