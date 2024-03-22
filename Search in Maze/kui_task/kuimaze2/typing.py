from enum import IntEnum
from typing import Mapping, Sequence

from kuimaze2.map import State, Action


Path = list[State]
"""Path type annotation: a sequence of states. May be useful for solvers."""

ActionValues = Mapping[Action, float]
"""ActionValues type annotation: representation of a distribution over actions."""

VTable = Mapping[State, float]
"""VTable type annotation: a tabular representation of V function. May be useful for solvers."""

QTable = Mapping[State, ActionValues]
"""QTable type annotation: a tabular representation of Q function. May be useful for solvers."""

Policy = Mapping[State, Action]
"""Policy type annotation: a representation of a policy, the return type of MDP and RL problems."""

Rewards = Mapping[str, float]
"""Rewards type annotation: rewards for states of 3 types: "goal", "danger", and "normal"."""
