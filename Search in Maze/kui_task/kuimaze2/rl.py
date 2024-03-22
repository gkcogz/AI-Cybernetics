import random
from typing import Optional

from kuimaze2.mdp import MDP, Action, State, ActionValues
from kuimaze2.exceptions import NeedsResetError


class RLProblem:
    """Fully observable RL problem, i.e., the observation is the actual state."""

    def __init__(self, mdp: MDP):
        self.mdp = mdp
        self.current_state = next(iter(mdp.goals))
        self.last_step_performed = True

    def reset(self, /, state: Optional[State] = None, random_start: bool = False):
        """Reset the environment, return a new random initial (non-terminal) state for an episode.

        You can specify the initial state for testing purposes.
        """
        self.last_step_performed = False
        if state:
            self.current_state = state
        elif random_start:
            self.current_state = random.choice(self.mdp.get_states())
        else:
            self.current_state = self.mdp.map.start
        if not self.current_state:
            raise ResetImpossibleError("RLProblem: Unable to set new start state. "
                "Specify a particular state, define a start state on a map, or set random_start to True.")
        return self.current_state

    def get_action_space(self):
        """Return the union of all actions applicable in any state (except the EXIT action)"""
        return [action for action in Action]

    def sample_action(self, action_probs: Optional[ActionValues] = None):
        """Return a random action from the action space"""
        if not action_probs:
            return random.choice(self.get_action_space())
        assert sum(action_probs.values()) == 1
        return random.choices(
            list(action_probs.keys()),
            weights=list(action_probs.values())
        )[0]        # Intentional: take the first item, random.choices() always returns a list

    def step(self, action: Action):
        """Take a single step in the environment"""
        if self.last_step_performed:
            raise NeedsResetError("RLProblem: Episode terminated. You must call reset() again.")
        # If we are going to make a step from terminal, then we are done
        if self.mdp.is_terminal(self.current_state):
            self.last_step_performed = True
        reward = self.mdp.get_reward(self.current_state)
        self.current_state = self.mdp.get_transition_result(self.current_state, action)
        return self.current_state, reward, self.last_step_performed