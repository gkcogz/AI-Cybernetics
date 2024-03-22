import pytest
from kuimaze2 import SearchProblem, State, Action

class TestSearchProblem:

    def test_get_start_goals(self):
        env = SearchProblem.from_string("...\n.S.")
        assert env.get_start() == State(1, 1)
        assert env.get_goals() == set()

    def test_getActions(self):
        env = SearchProblem.from_string("S")
        assert env.get_actions(State(0, 0)) == [Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT]

    def test_getActions_returnsEmptyList_forWalls(self):
        env = SearchProblem.from_string("...\n.#.\n...")
        assert env.get_actions(State(1, 1)) == []

    def test_getTransitionResult_actionsPossible(self):
        env = SearchProblem.from_string("...\n.S.\n...")
        state = env.get_start()
        assert env.get_transition_result(state, Action.UP) == (State(0, 1), 1)
        assert env.get_transition_result(state, Action.RIGHT) == (State(1, 2), 1)
        assert env.get_transition_result(state, Action.DOWN) == (State(2, 1), 1)
        assert env.get_transition_result(state, Action.LEFT) == (State(1, 0), 1)

    def test_getTransitionResult_actionsImpossible(self):
        env = SearchProblem.from_string("S")
        state = env.get_start()
        assert env.get_transition_result(state, Action.UP) == (State(0, 0), 1)
        assert env.get_transition_result(state, Action.RIGHT) == (State(0, 0), 1)
        assert env.get_transition_result(state, Action.DOWN) == (State(0, 0), 1)
        assert env.get_transition_result(state, Action.LEFT) == (State(0, 0), 1)

