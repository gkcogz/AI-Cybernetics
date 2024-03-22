from dataclasses import astuple, dataclass
import tkinter as tk
from typing import Mapping, Self, Callable


from kuimaze2.map import Border, Map, State, Role, Cell

MAX_WIN_WIDTH = 800
MAX_WIN_HEIGHT = 800
SQUARE_SIZE = 100
MARGIN_SIZE = 50
CIRCLE_DIAMETER = 0.5  # Relative to SQUARE_SIZE
FONT_FAMILY = "Helvetica"


@dataclass(frozen=True)
class Color:
    r: int
    g: int
    b: int

    def mix(self, other: Self, factor: float) -> Self:
        return Color(
            int(self.r * (1 - factor) + other.r * factor),
            int(self.g * (1 - factor) + other.g * factor),
            int(self.b * (1 - factor) + other.b * factor),
        )

    def to_hex(self) -> str:
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"


COLOR_FROM_ROLE: Mapping[Role, Color] = {
    Role.EMPTY: Color(255, 255, 255),   # "#ffffff",
    Role.WALL: Color(0, 0, 0),          # "#000000",
    Role.START: Color(0, 0, 255).mix(Color(255, 255, 255), 0.5),       # "#0000ff",
    Role.GOAL: Color(0, 255, 0).mix(Color(255, 255, 255), 0.5),        # "#00ff00",
    Role.DANGER: Color(255, 0, 0).mix(Color(255, 255, 255), 0.5),      # "#ff0000",
}
DIVIDER_COLOR = Color(204, 204, 204)    # "#cccccc"
CURRENT_CIRCLE_COLOR = Color(255, 127, 127)
NEXT_CIRCLE_COLOR = Color(255, 255, 0)
FRONTIER_CIRCLE_COLOR = Color(224, 224, 224)


def create_color_mixer(color_from_value: tuple[tuple[float, Color]]) -> Callable[[float], Color]:
    def color_of_value(value: float) -> Color:
        if value <= color_from_value[0][0]:
            return color_from_value[0][1]
        if value >= color_from_value[-1][0]:
            return color_from_value[-1][1]
        min_value, color1 = color_from_value[0]
        max_value, color2 = color_from_value[-1]
        factor = (value - min_value) / (max_value - min_value)
        return color1.mix(color2, factor)
    return color_of_value

@dataclass(frozen=True)
class RectCoords:
    left: int
    top: int
    right: int
    bottom: int
    
    def __iter__(self):
        return iter(astuple(self))
    
    def of_border_line(self, border: Border):
        match border:
            case Border.TOP:
                return self.left, self.top, self.right, self.top
            case Border.RIGHT:
                return self.right, self.top, self.right, self.bottom
            case Border.BOTTOM:
                return self.left, self.bottom, self.right, self.bottom
            case Border.LEFT:
                return self.left, self.top, self.left, self.bottom



class SquareCanvas(tk.Canvas):
    
    def __init__(self,
                 parent: tk.Tk,
                 map: Map,
                 sq_size: int = 0,
                 **kwargs):
        if sq_size == 0:
            max_sq_width = (MAX_WIN_WIDTH - 2 * MARGIN_SIZE) // map.width
            max_sq_height = (MAX_WIN_HEIGHT - 2 * MARGIN_SIZE) // map.height
            sq_size = min([max_sq_width, max_sq_height, SQUARE_SIZE])
        canvas_width = sq_size * map.width + 2 * MARGIN_SIZE
        canvas_height = sq_size * map.height + 2 * MARGIN_SIZE
        super().__init__(parent, width=canvas_width, height=canvas_height, **kwargs)
        self.map: Map = map
        self.squares = {}
        self.circles = {}
        self.texts = {}
        self.path_line = None
        self.square_size = sq_size
        self.font_size = max(2, int(0.2 * self.square_size))
        self.draw_map()


    def color_from_value(self, value: float):
        return Color(255, 255, 255)

    def draw_map(self):
        self.draw_squares()
        self.draw_row_indices()
        self.draw_col_indices()
        self.draw_circles()
        self.draw_square_texts()
        
    def draw_squares(self):
        """Draw all squares in the map, parts that can change color based on value"""
        for sq in self.map:
            self.draw_square(sq)
            
    def draw_square(self, square: Cell):
        coords = self._coords_from_position(square.position)
        self.squares[square.position] = self.create_rectangle(
            *coords,
            fill=COLOR_FROM_ROLE[square.role].to_hex(),
            outline=DIVIDER_COLOR.to_hex(),
            width=3)
        self._draw_border(square.border, coords)
        
    def _coords_from_position(self, pos: State) -> RectCoords:
        left = MARGIN_SIZE + self.square_size * pos.c
        top = MARGIN_SIZE + self.square_size * pos.r
        return RectCoords(left, top, left+self.square_size, top+self.square_size)

    def _center_from_position(self, pos: State) -> tuple[int, int]:
        rect = self._coords_from_position(pos)
        return int((rect.left+rect.right)/2), int((rect.top+rect.bottom)/2)
    
    def _draw_border(self, square_border: Border, coords: RectCoords):
        color = COLOR_FROM_ROLE[Role.WALL].to_hex()
        for border in Border:
            if border in square_border:
                line_coords = coords.of_border_line(border)
                self.create_line(*line_coords, fill=color, width=5, capstyle="round")
                
    def draw_row_indices(self):
        for row in range(self.map.height):
            self.create_text(MARGIN_SIZE//2, 
                             MARGIN_SIZE + self.square_size * row + self.square_size//2, 
                             text=str(row),
                             font=(FONT_FAMILY, self.font_size))
            self.create_text(MARGIN_SIZE + self.square_size * self.map.width + MARGIN_SIZE//2, 
                             MARGIN_SIZE + self.square_size * row + self.square_size//2, 
                             text=str(row),
                             font=(FONT_FAMILY, self.font_size))

    def draw_col_indices(self):
        for col in range(self.map.width):
            self.create_text(MARGIN_SIZE + self.square_size * col + self.square_size//2, 
                             MARGIN_SIZE//2, 
                             text=str(col),
                             font=(FONT_FAMILY, self.font_size))
            self.create_text(MARGIN_SIZE + self.square_size * col + self.square_size//2, 
                             MARGIN_SIZE + self.square_size * self.map.height + MARGIN_SIZE//2, 
                             text=str(col),
                             font=(FONT_FAMILY, self.font_size))

    def draw_circles(self):
        """Draw all squares in the map, parts that can change color based on value"""
        for sq in self.map:
            self.draw_circle(sq)

    def draw_circle(self, square: Cell):
        coords = self._coords_from_position(square.position)
        self.circles[square.position] = self.create_oval(
            *coords, fill='', outline='', state=tk.HIDDEN,
        )

    def draw_square_texts(self, texts=None, default_text=""):
        texts = texts or {}
        for sq in self.map:
            text = texts.get(sq.position, default_text)
            coords = self._coords_from_position(sq.position)
            left = coords.left + self.square_size//2
            top = coords.top + self.square_size//2
            self.texts[sq.position] = self.create_text(left, top, text=text, font=(FONT_FAMILY, self.font_size))

    def draw_path(self, path: list[State]):
        if self.path_line:
            self.delete(self.path_line)
        points = []
        for state in path:
            sq_center = self._center_from_position(state)
            points.extend(sq_center[:])
        self.path_line = self.create_line(points, fill="red", width=7, capstyle="round", arrow=tk.LAST, arrowshape=(12, 20, 6))

    def modify_square_color(self, position: State, color: Color):
        self.itemconfig(self.squares[position], fill=color.to_hex())

    def modify_circle_color(self, position: State, color: Color, visible: bool = True):
        self.itemconfig(self.circles[position], fill=color.to_hex(), state=tk.NORMAL if visible else tk.HIDDEN)
        
    def modify_circle_visibility(self, position: State, visible: bool = True):
        self.itemconfig(self.circles[position], state=tk.NORMAL if visible else tk.HIDDEN)

    def modify_square_text(self, position: State, text: str):
        self.itemconfig(self.texts[position], text=text)

    def update_square_texts(self, texts: dict[State, str]):
        for position, text in texts.items():
            self.modify_square_text(position, text)

    def set_square_colors(self, colors: dict[State, Color], keep_role_colors=False):
        for position, color in colors.items():
            if keep_role_colors and self.map[position].role != Role.EMPTY:
                continue
            self.modify_square_color(position, color)


class SearchCanvas(SquareCanvas):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_colors = (
            (0, COLOR_FROM_ROLE[Role.START]),
            (self.map.number_of_accessible_states, Color(255,255,255)),
        )
        self.current_circle = None
        self.next_circles = []
        self.frontier_circles = []

    def color_from_value(self, value: float = None, factor: float = None) -> Color:
        ((min_value, color1), (max_value, color2)) = self.value_colors
        if not factor:
            if value <= min_value:
                return color1
            if value >= max_value:
                return color2
            factor = (value - min_value) / (max_value - min_value)
        color = color1.mix(color2, factor)
        return color

    def set_square_colors_from_values(self, colors: dict[State, float]):
        colors = {position: self.color_from_value(value) for position, value in colors.items()}
        self.set_square_colors(colors, keep_role_colors=True)

    def set_square_colors_from_visited(self, visited):
        color = COLOR_FROM_ROLE[Role.START].mix(Color(255, 255, 255), 0.5)
        colors = {position: color for position in visited}
        self.set_square_colors(colors, keep_role_colors=True)

    def set_current_state(self, state: State = None):
        if self.current_circle:
            self.modify_circle_visibility(self.current_circle, False)
        self.current_circle = state
        if state:
            self.modify_circle_color(state, CURRENT_CIRCLE_COLOR)
            self.modify_circle_visibility(state, True)

    def set_next_states(self, next_states: list[State] = []):
        self.reset_next_states()
        for state in next_states:
            self.add_next_state(state)

    def add_next_state(self, state: State):
        self.modify_circle_color(state, NEXT_CIRCLE_COLOR)
        self.modify_circle_visibility(state, True)
        self.next_circles.append(state)

    def reset_next_states(self):
        for state in self.next_circles:
            if state in self.frontier_circles:
                self.modify_circle_color(state, FRONTIER_CIRCLE_COLOR)
            else:
                self.modify_circle_visibility(state, False)
        self.next_circles = []

    def set_frontier_states(self, frontier):
        self.reset_frontier_states()
        for state in frontier:
            self.add_frontier_state(state)

    def add_frontier_state(self, state: State):
        self.modify_circle_color(state, FRONTIER_CIRCLE_COLOR)
        self.modify_circle_visibility(state, True)
        self.frontier_circles.append(state)

    def reset_frontier_states(self):
        for state in self.frontier_circles:
            self.modify_circle_visibility(state, False)
        self.frontier_circles = []


class ValueCanvas(SquareCanvas):

    def __init__(self, parent: tk.Tk, map: Map, sq_size: int = 0, **kwargs):
        if sq_size == 0:
            # Allow for 2 canvases above each other
            max_sq_width = (MAX_WIN_WIDTH - 2 * MARGIN_SIZE) // map.width
            max_sq_height = (MAX_WIN_HEIGHT - 4 * MARGIN_SIZE) // (2*map.height)
            sq_size = min([max_sq_width, max_sq_height, SQUARE_SIZE])
        print(f"Value canvas size: {sq_size}")
        super().__init__(parent=parent, map=map, sq_size=sq_size, **kwargs)


class QValueCanvas(SquareCanvas):

    def __init__(self, parent: tk.Tk, map: Map, sq_size: int = 0, **kwargs):
        if not sq_size:
            # Allow for 2 canvases above each other
            max_sq_width = (MAX_WIN_WIDTH - 2 * MARGIN_SIZE) // map.width
            max_sq_height = (MAX_WIN_HEIGHT - 2 * MARGIN_SIZE) // (2*map.height)
            sq_size = min([max_sq_width, max_sq_height, SQUARE_SIZE])
        print(f"QValue canvas size: {sq_size}")
        super().__init__(parent=parent, map=map, sq_size=sq_size, **kwargs)
