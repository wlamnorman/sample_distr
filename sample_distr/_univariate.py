from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable


@dataclass()
class UniVarSampleDistr[T: float | int]:
    support_weights: dict[T, float] = field(default_factory=lambda: defaultdict(float))
    total_weight: float = 0.0

    def observe(self, observation: T, weight: float = 1.0):
        self.support_weights[observation] += weight
        self.total_weight += weight

    def update_from_other(self, other: UniVarSampleDistr):
        for supp, supp_w in other.support_weights.items():
            self.observe(supp, supp_w)

    def get_support(self) -> list[T]:
        return sorted(list(self.support_weights.keys()))

    def get_probabilities(self) -> list[float]:
        return [self.eq(x) for x in self.get_support()]

    def _prob_from_element_and_operator(
        self,
        element: T,
        operator_: Callable[[T, T], bool],
    ) -> float:
        weight = 0.0
        for comparsion_element, comparsion_weight in self.support_weights.items():
            if operator_(comparsion_element, element):
                weight += comparsion_weight

        return weight / self.total_weight

    def eq(self, element: T) -> float:
        """`P(X=element)`"""
        return self._prob_from_element_and_operator(element, lambda x, y: x == y)

    def le(self, element: T) -> float:
        """`P(X<=element)`"""
        return self._prob_from_element_and_operator(element, lambda x, y: x <= y)

    def lt(self, element: T) -> float:
        """`P(X<element)`"""
        return self._prob_from_element_and_operator(element, lambda x, y: x < y)

    def ge(self, element: T) -> float:
        """`P(X>=element)`"""
        return 1.0 - self.lt(element)

    def gt(self, element: T) -> float:
        """`P(X>element)`"""
        return 1.0 - self.le(element)
