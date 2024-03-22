from kuimaze2.map import State, Action

class TestState:

    def test_creation(self):
        p = State(r=1, c=2)
        assert p[0] == 1
        assert p.r == 1
        assert p[1] == 2
        assert p.c == 2
        
    def test_addition(self):
        s = State(1, 2)
        a = Action.RIGHT
        s2 = s + a.to_vec()
        assert s2 == State(1, 3)
        
    def test_len(self):
        s = State(1, 2)
        assert len(s) == 2
        
    def test_equality(self):
        s1 = State(1, 2)
        s2 = State(1, 2)
        assert s1 == s2


