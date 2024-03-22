from typing import Mapping, Optional
from abc import ABC, abstractmethod
from enum import IntEnum
import random
from kuimaze2.map import Map, State, Action
from kuimaze2.typing import Rewards, ActionValues


DEFAULT_REWARDS: Rewards = {
    "goal": 1.0,
    "danger": -1.0,
    "normal": -0.04,
}


class Confusion(IntEnum):
    NONE = 0
    RIGHT = 1
    BACKWARD = 2
    LEFT = 3

    def apply_to(self, action: Action) -> Action:
        return Action((action + self) % 4)


class ActionsModel(ABC):

    @abstractmethod
    def get_actions_probs(self, action: Action) -> ActionValues:
        ...

    @abstractmethod
    def sample_action(self, action: Action) -> Action:
        ...


class DeterministicActions(ActionsModel):

    def get_actions_probs(self, action: Action) -> ActionValues:
        return {action: 1.0}

    def sample_action(self, action: Action) -> Action:
        return action


class StochasticActions(ActionsModel):

    def __init__(self, forward: float, left: float, right: float, backward: float):
        assert forward + left + right + backward == 1.0
        self.confusion_probs = {
            Confusion.NONE: forward,
            Confusion.RIGHT: right,
            Confusion.BACKWARD: backward,
            Confusion.LEFT: left
        }

    def _sample_confusion(self) -> Confusion:
        return random.choices(
            list(self.confusion_probs.keys()),
            weights=list(self.confusion_probs.values()),
        )[0]

    def get_actions_probs(self, action: Action) -> ActionValues:
        return {confusion.apply_to(action): prob for confusion, prob in self.confusion_probs.items()}

    def sample_action(self, action: Action) -> Action:
        confusion = self._sample_confusion()
        return confusion.apply_to(action)



class MDP:
    """MDP problem class defined over a map with deterministic or stochastic actions."""

    def __init__(
            self,
            map: Map,
            actions_model: Optional[ActionsModel] = None,
            rewards: Optional[Rewards] = None,
    ):
        self.map = map
        self.actions_model = actions_model or DeterministicActions()
        self.rewards = rewards or DEFAULT_REWARDS
        # MDP may redefine goal and danger state in the future
        # Right now, just use those in the map
        self.goals = self.map.goals
        self.dangers = self.map.dangers

    @classmethod
    def from_string(cls, map: str, *args, **kwargs) -> "MDPProblem":
        return cls(Map.from_string(map), *args, **kwargs)

    def get_states(self) -> list[State]:
        """Return all states in the MDP problem."""
        return [cell.position for cell in self.map if cell.is_free()]

    def get_non_terminal_states(self) -> list[State]:
        return [state for state in self.get_states() if not self.is_terminal(state)]

    def get_actions(self, state: State) -> list[Action]:
        return list(Action)

    def get_reward(self, state: State) -> float:
        """Return reward for leaving a state."""
        if self.is_goal(state):
            return self.rewards["goal"]
        if self.is_danger(state):
            return self.rewards["danger"]
        return self.rewards["normal"]

    def get_transition_result(self, state: State, action: Action) -> State:
        """Apply a DETERMINISTIC action to a state."""
        if self.is_terminal(state):
            return state
        return self.map.get_transition_result(state, action)

    def get_next_states_and_probs(self, state: State, action: Action) -> list[tuple[State, float]]:
        actions_probs = self.actions_model.get_actions_probs(action)
        return [(self.get_transition_result(state, action), prob) for action, prob in actions_probs.items()]

    def is_terminal(self, state: State) -> bool:
        return self.is_goal(state) or self.is_danger(state)

    def is_goal(self, state: State) -> bool:
        return state in self.goals

    def is_danger(self, state: State) -> bool:
        return state in self.dangers


class MDPProblem(MDP):
    pass

