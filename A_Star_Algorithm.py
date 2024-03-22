# An Introduction to A* Algorithm
# The code is originally written by Tech with Tim.

import pygame
import math

# Variant of Queue that retrieves open entries in priority order (lowest first).
# Entries are typically tuples of the form:  (priority number, data).

from queue import PriorityQueue

# help(PriorityQueue)
# Set variable WIDTH. The most essential variable.

WIDTH = 800

# It is a cube.
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption('A* Path Finding Algorithm by TTT, rewritten by Oguz Can Gökce')


RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 255, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)

# 57:13
# 22:57

# We defined a class.
# You can also call it 'Node'.
class Spot:
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col

        # We track the actual coordinates of the point.
        self.x = row * width
        self.y = col * width 
        self.color = WHITE
        self.neighbors = []
        self.width = width
        self.total_rows = total_rows

    def get_pos(self):

        # row x col
        return self.row, self.col
    

    # If the cube is red, we already looked for it.
    # If it is black, then it is a barrier.
    # Defining, descriptive functions at the start.

    def is_closed(self):
        return self.color == RED
    
    # Unexplored cube is green.
    def is_open(self):
        return self.color == GREEN
    
    # Barrier.
    def is_barrier(self):
        return self.color == BLACK
    
    # Start point.
    def is_start(self):
        return self.color == ORANGE
    
    def is_end(self):
        return self.color == TURQUOISE
    
    # This is an executive function, so you should code it as in the 
    # following. It is not a defining function like above as part of class 
    # Spot.
    def reset(self):
        self.color = WHITE

    # Executive functions. Setting variables.
        
    def make_start(self):
        self.color = ORANGE

    def make_closed(self):
         self.color = RED
    
    def make_open(self):
         self.color = GREEN

    def make_barrier(self):
         self.color = BLACK

    def make_end(self):
        self.color = TURQUOISE

    def make_path(self):
         self.color = PURPLE

    # Where do we wanna draw the window? 
        
    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))
    
    # grid is not defined yet.
    # 'pass' allows you to pass a function temporarily.
    def update_neighbors(self, grid):
        self.neighbors = []
        # If the row number smaller than the border and the cube no. is not defined as barrier,
        # add it to the list of neighbors.
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier(): # DOWN
            self.neighbors.append(grid[self.row + 1][self.col])
        
        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier(): # UP
            self.neighbors.append(grid[self.row - 1][self.col]) 

        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier(): # RIGHT
            self.neighbors.append(grid[self.row][self.col + 1])
        
        if self.row > 0 and not grid[self.row][self.col - 1].is_barrier(): # LEFT
            self.neighbors.append(grid[self.row][self.col - 1])

    # lt stands for less than, essentially this is how we handle what happens
    # if we compare two spots together.
    # other is not defined yet.

    def __lt__(self, other):
        return False


# Manhattan Distance
def h(p1, p2):

    # It is a shortcut in Python.
    x1, y1 = p1
    x2, y2 = p2

    # abs is a built-in function.
    return abs(x1 - x2) + abs(y1- y2)

def reconstruct_path(came_from, current, draw):
    while current in came_from:
        current = came_from[current]
        current.make_path()
        draw()

    # Lambda allows you to call pass a function into the main function,
    # for example. 

    # 1:24:00
def algorithm(draw, grid, start, end):

    # Start node with F score = 0.
    # F() = G() + H()

    count = 0
    open_set = PriorityQueue()

    # We need to check if there something in the queue or not.
    # API for PriorityQueue.
    # Tie-breaker.
    # (F-score, the count, the node)

    open_set.put((0, count, start))  

    # Keeps track of which node came from where.
    came_from = {}

    g_score = {spot: float("inf") for row in grid for spot in row}
    g_score[start] = 0

    f_score = {spot: float("inf") for row in grid for spot in row}
    f_score[start] = h(start.get_pos(), end.get_pos())


    open_set_hash = {start}

    # After we considered every single possible node.
    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = open_set.get()[2]
        open_set_hash.remove(current)

        if current == end:
            reconstruct_path(came_from, end, draw)
            end.make_end()
            return True
        
        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1

            # If the current g_socre is lower than the previous one,
            # update the score.
            
            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos())
                # So, we can check if the neighbor is in the hast or not.
                # Thus, we prevent overriding same cube couple times.
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()

        draw()

                # We already considered that node and we add to red nodes.
        if current != start:
            current.make_closed()

    return False



# Width is the width of our entire grid.

def make_grid(rows, width):
    grid = []


    #// is integer division.
    # What will be the size of cubes.
    gap = width // rows
    for i in range(rows):


        # We add lists to into the list 'grid'.
        grid.append([])
        for j in range(rows):

            # reminder. def __init__(self, row, col, width, total_rows).
            spot = Spot(i, j, gap, rows)
            grid[i].append(spot)
    
    return grid

# Let's draw the cude. We do not have grid lines.

def draw_grid(win, rows, width):
    gap = width // rows
    for i in range(rows):
        # Where should we draw the grid lines.
        pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
        for j in range(rows):
            pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))




# What cube did we click on? To determine the mouse position.
def draw(win, grid, rows, width): 
    win.fill(WHITE)

    for row in grid:
        for spot in row:
            spot.draw(win)

    draw_grid(win, rows, width)
    pygame.display.update()

# 46:53
# Since, it a cube, we only defined width. 
# And later, we derived row and col variables 

def get_clicked_pos(pos, rows, width):
    gap = width // rows
    y, x = pos

    row = y // gap
    col = x // gap

    return row, col

# Main loop:
def main(win, width):

    # You can change the number to whatever you want.
    # The second most essential variable: row and column no, which we obtain
    # with get_clicked_pos() from Pygame. 
    ROWS = 50
    grid = make_grid(ROWS, width)

    # Start and end spots are defined as none.
    # None is percieved as False (Boolean).
    start = None
    end = None

    run = True

    while run:
        draw(win, grid, ROWS, width)



        # Whatever event happens in pygame.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # If we press that 'x' button exit the game.
                # run is set to False.
                run = False


            # 0,1,2 left-mid-right button.
            if pygame.mouse.get_pressed()[0]: # LEFT
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                spot = grid[row][col]
            

                # if, elif, elif state an order of execution.
                # Set the first spot to start.
                if not start:
                    start = spot
                    start.make_start()
                
                # Set the second spot to start,
                elif not end:
                    end = spot
                    end.make_end()

                elif spot != start and spot != end:
                    spot.make_barrier()
                    
            elif pygame.mouse.get_pressed()[2]: # RIGHT
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                spot = grid[row][col]
                spot.reset()
                if spot == start:
                    start = None

                elif spot == end:
                    end = None

            # Loop.
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start and end:
                    for row in grid:
                        for spot in row:
                            spot.update_neighbors(grid)

                    # Once, we start we will call algorithm function
                    # Lambda is an anonymous function. 
                    # Let' say x = Lambda: print('Hello!')
                    # If you call x(), then you call the function embedded into lambda.
                    # Thus, lambda allows you to pass a function into another function.
                    

                    algorithm(lambda: draw(win, grid, ROWS, width), grid, start, end)



    # First and second time, you specify the start and the end state,
    # if continues it specifies the position of barries marked with grey.
                
                # To clean the entire screen.
                if event.key == pygame.K_c:
                    start = None
                    end = None
                    grid = make_grid(ROWS, width)


    pygame.quit()

main(WIN, WIDTH)

# We did so far the visualziation part of the game.






    






























    
    













