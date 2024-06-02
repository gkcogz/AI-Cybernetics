import pytest
from kuimaze2.map import Border, Action, State, Role, Cell, Map, MazeVec


class TestMapCreation:

    def test_creation_empty_map(self):
        map = Map()
        assert isinstance(map, Map)
        assert map.width == 1
        assert map.height == 1
        assert map[State(0, 0)].border == Border.TOP | Border.RIGHT | Border.BOTTOM | Border.LEFT

    def test_creation_single_square(self):
        map = Map(cells=[Cell()])
        assert isinstance(map, Map)
        assert map.width == 1
        assert map.height == 1
        assert map[State(0, 0)] == Cell(
            State(0, 0),
            Role.EMPTY,
            Border.TOP | Border.RIGHT | Border.BOTTOM | Border.LEFT,
        )

    def test_creation_single_wall(self):
        map = Map(cells=[Cell(role=Role.WALL)])
        assert isinstance(map, Map)
        assert map.width == 1
        assert map.height == 1
        assert map[State(0, 0)] == Cell(
            State(0, 0),
            Role.WALL,
            Border.TOP | Border.RIGHT | Border.BOTTOM | Border.LEFT,
        )

    def test_creation_two_empty_squares_horizontal(self):
        map = Map(cells=[Cell(State(0, 0)), Cell(State(0, 1))])
        assert isinstance(map, Map)
        assert map.width == 2
        assert map.height == 1
        assert map[State(0, 0)] == Cell(
            State(0, 0),
            Role.EMPTY,
            Border.TOP | Border.BOTTOM | Border.LEFT,
        )
        assert map[State(0, 1)] == Cell(
            State(0, 1),
            Role.EMPTY,
            Border.TOP | Border.RIGHT | Border.BOTTOM,
        )

    def test_creation_two_empty_squares_vertical(self):
        map = Map(cells=[Cell(State(0, 0)), Cell(State(1, 0))])
        assert isinstance(map, Map)
        assert map.width == 1
        assert map.height == 2
        assert map[State(0, 0)] == Cell(
            State(0, 0),
            Role.EMPTY,
            Border.TOP | Border.RIGHT | Border.LEFT,
        )
        assert map[State(1,0)] == Cell(
            State(1, 0),
            Role.EMPTY,
            Border.LEFT | Border.RIGHT | Border.BOTTOM,
        )

    def test_creation_two_by_two_single_empty_square(self):
        map = Map(cells=[Cell(State(1, 1))])
        assert isinstance(map, Map)
        assert map.width == 2
        assert map.height == 2
        assert map[State(0, 0)] == Cell(
            State(0, 0), 
            Role.WALL, 
            Border.TOP | Border.LEFT | Border.BOTTOM | Border.RIGHT) 
        assert map[State(1, 0)] == Cell(
            State(1, 0),
            Role.WALL,
            Border.TOP | Border.LEFT | Border.BOTTOM | Border.RIGHT,
        )
        assert map[State(0, 1)] == Cell(
            State(0, 1),
            Role.WALL,
            Border.TOP | Border.LEFT | Border.BOTTOM | Border.RIGHT,
        )
        assert map[State(1, 1)] == Cell(
            State(1, 1),
            Role.EMPTY,
            Border.TOP | Border.LEFT | Border.BOTTOM | Border.RIGHT,
        )


class TestMapGoalsAndDangers:

    def test_goals_noGoals(self):
        map = Map.from_string("S")
        assert map.goals == set()

    def test_goals_G(self):
        map = Map.from_string("G")
        assert map.goals == {State(0, 0)}

    def test_goals_GG(self):
        map = Map.from_string("GG")
        assert map.goals == {State(0, 0), State(0, 1)}

    def test_goals_noDangers(self):
        map = Map.from_string("S")
        assert map.goals == set()

    def test_dangers_D(self):
        map = Map.from_string("D")
        assert map.dangers == {State(0, 0)}

    def test_dangers_DD(self):
        map = Map.from_string("DD")
        assert map.dangers == {State(0, 0), State(0, 1)}
