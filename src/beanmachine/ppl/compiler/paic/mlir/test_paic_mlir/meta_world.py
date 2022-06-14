import typing
from beanmachine.ppl import RVIdentifier

class World:
    def __init__(self, observations: typing.Dict[RVIdentifier, float], queries:typing.List[RVIdentifier]):
        self.queries = queries
        self.observations = observations

    def log_prob_of(self, query:RVIdentifier, value:float) -> float:
        raise NotImplementedError("Compiler should be generating this call")

    def set_value(self, rv:RVIdentifier, value:float):
        raise NotImplementedError("Compiler should be generating this call")