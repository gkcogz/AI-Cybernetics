import pytest
from collections import Counter

from kuimaze2 import RLProblem, State, Action
from kuimaze2.mdp import MDP
from kuimaze2.exceptions import NeedsResetError

def test_creation():
    mdp = MDP.from_string("G")
    problem = RLProblem(mdp)
    assert problem

def test_reset():
    mdp = MDP.from_string("SG")
    problem = RLProblem(mdp)
    init_state = problem.reset(random_start=True)
    assert init_state in mdp.get_states()

def test_get_action_space():
    mdp = MDP.from_string("G")
    problem = RLProblem(mdp)
    actions = problem.get_action_space()
    assert len(actions) == 4
    assert Action.UP in actions
    assert Action.RIGHT in actions
    assert Action.DOWN in actions
    assert Action.LEFT in actions

class TestSampleAction:
    def test_sample_action_uniformly(self):
        # This is a stochastic test. Not good.
        # How to test that sample_action returns all possible actions uniformly?
        mdp = MDP.from_string("SG")
        problem = RLProblem(mdp)
        init_state = problem.reset()
        action = problem.sample_action()
        assert action in problem.get_action_space()

    def test_sample_action_uniformly_action_probs(self):
        # This is a stochastic test. Not good.
        # How to test that sample_action returns all possible actions uniformly?
        mdp = MDP.from_string("SG")
        problem = RLProblem(mdp)
        init_state = problem.reset()
        action = problem.sample_action({Action.UP: 0.25, Action.RIGHT: 0.25, Action.DOWN: 0.25, Action.LEFT: 0.25})
        assert action in problem.get_action_space()

    def test_sample_action_UP(self):
        mdp = MDP.from_string("SG")
        problem = RLProblem(mdp)
        init_state = problem.reset()
        action = problem.sample_action(Counter({Action.UP: 1}))
        assert action == Action.UP

    def test_sample_action_UP_multi(self):
        mdp = MDP.from_string("SG")
        problem = RLProblem(mdp)
        init_state = problem.reset()
        action = problem.sample_action(
            {Action.UP: 1, Action.RIGHT: 0, Action.DOWN: 0, Action.LEFT: 0})
        assert action == Action.UP

    def test_sample_action_RIGHT(self):
        mdp = MDP.from_string("SG")
        problem = RLProblem(mdp)
        init_state = problem.reset()
        action = problem.sample_action(Counter({Action.RIGHT: 1}))
        assert action == Action.RIGHT

    def test_sample_action_DOWN(self):
        mdp = MDP.from_string("SG")
        problem = RLProblem(mdp)
        init_state = problem.reset()
        action = problem.sample_action(Counter({Action.DOWN: 1}))
        assert action == Action.DOWN

    def test_sample_action_LEFT(self):
        mdp = MDP.from_string("SG")
        problem = RLProblem(mdp)
        init_state = problem.reset()
        action = problem.sample_action(Counter({Action.LEFT: 1}))
        assert action == Action.LEFT


class TestStep:

    def test_terminal_state(self):
        mdp = MDP.from_string("G")
        problem = RLProblem(mdp)
        with pytest.raises(NeedsResetError):
            new_state, reward, terminated = problem.step(Action.UP)

    def test_nonterminal_state(self):
        mdp = MDP.from_string("S.G", rewards={"goal": 10, "danger": -10, "normal": -5})
        problem = RLProblem(mdp)
        init_state = problem.reset(State(0, 0))
        assert init_state == State(0, 0)
        new_state, reward, terminated = problem.step(Action.RIGHT)
        assert new_state == State(0, 1)
        assert reward == -5
        assert terminated == False

    def test_reaching_goal(self):
        mdp = MDP.from_string("SG", rewards={"goal": 10, "danger": -10, "normal": -5})
        problem = RLProblem(mdp)
        init_state = problem.reset(state=State(0, 0))
        assert init_state == State(0, 0)
        new_state, reward, terminated = problem.step(Action.RIGHT)
        assert new_state == State(0, 1)
        assert reward == -5 # Include also reward for reaching the goal state
        assert terminated == False
        new_state, reward, terminated = problem.step(Action.RIGHT)
        assert new_state == State(0, 1)
        assert reward == 10  # Include also reward for reaching the goal state
        assert terminated == True
