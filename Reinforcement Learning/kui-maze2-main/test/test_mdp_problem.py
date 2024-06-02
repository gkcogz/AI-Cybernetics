import pytest

from kuimaze2 import MDPProblem, State, Action
from .mdp_fixtures import mdp_G, mdp_SG, mdp_3x3, mdp_3x3_impossible


class TestMDPProblem:

    def test_get_states(self, mdp_G):
        assert mdp_G.get_states() == [State(0, 0)]

    def test_getNextStatesAndProbs_allActionsPossible(self, mdp_3x3):
        mdp = mdp_3x3
        assert mdp.get_next_states_and_probs(State(1, 1), Action.UP) == [(State(0, 1), 1.0)]
        assert mdp.get_next_states_and_probs(State(1, 1), Action.RIGHT) == [(State(1, 2), 1.0)]
        assert mdp.get_next_states_and_probs(State(1, 1), Action.DOWN) == [(State(2, 1), 1.0)]
        assert mdp.get_next_states_and_probs(State(1, 1), Action.LEFT) == [(State(1, 0), 1.0)]

    def test_getNextStatesAndProbs_noActionsPossible(self, mdp_3x3_impossible):
        mdp = mdp_3x3_impossible
        assert mdp.get_next_states_and_probs(State(1, 1), Action.UP) == [(State(1, 1), 1.0)]
        assert mdp.get_next_states_and_probs(State(1, 1), Action.RIGHT) == [(State(1, 1), 1.0)]
        assert mdp.get_next_states_and_probs(State(1, 1), Action.DOWN) == [(State(1, 1), 1.0)]
        assert mdp.get_next_states_and_probs(State(1, 1), Action.LEFT) == [(State(1, 1), 1.0)]

    def test_getNextStatesAndProbs_inTerminal(self, mdp_3x3):
        mdp = mdp_3x3
        assert mdp.get_next_states_and_probs(State(2, 2), Action.UP) == [(State(2, 2), 1.0)]
        assert mdp.get_next_states_and_probs(State(2, 2), Action.RIGHT) == [(State(2, 2), 1.0)]
        assert mdp.get_next_states_and_probs(State(2, 2), Action.DOWN) == [(State(2, 2), 1.0)]
        assert mdp.get_next_states_and_probs(State(2, 2), Action.LEFT) == [(State(2, 2), 1.0)]

    def test_isTerminal_returnsTrue_forGoal(self, mdp_SG):
        assert mdp_SG.is_terminal(State(0, 1)) == True

    def test_isTerminal_returnsFalse_forNonGoal(self, mdp_SG):
        assert mdp_SG.is_terminal(State(0, 0)) == False
