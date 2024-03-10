
# I decided to use heapq.
import heapq
import itertools
from kuimaze2 import SearchProblem, Map, State


# We define a new class named 'Agent'.
class Agent:

    # We initialize a new instance of the 'Agent', method #01.
    # 'environment is an instance of 'Search Problem'.
    def __init__(self, environment: SearchProblem):
        self.environment = environment

    # We use the Manhattan distance to calculate the h-score from any state
    # to the goal state.
    # The algorithm will continiously calculate f_score = h_score (heuristic) + g_score.
    
    def heuristic(self, state: State, goal: State) -> int:

        # We use .r and .c attributes defined in Map. Thus, we calculate
        # the distance.

        return abs(state.r - goal.r) + abs(state.c - goal.c)
    
    # Maintain an open list (priority queue) based on the f score
    # of each state and a closed list to track visited states.

    # Once the goal is reached, I reconstruct the path from goal
    # back to the start using the parent pointers I maintain during 
    # search.

    # We define a method to find and return the shortest path, method #02.
    def find_path(self) -> list[State]:
        
        # We retrieve the start state from the environment.
        start = self.environment.get_start()

        # We retrieve all the goal states.
        goals = self.environment.get_goals()

        '''Initializing Search'''

        # We check if 'goals' set is empty. 
        if not goals:
            raise ValueError("No goal state defined in the environment.")
        
        # Otherwise, we retrieve a goal state from the set.
        goal = next(iter(goals))

        # The open list is implemented as a priority queue.
        # Each entry is a tuple (f_score for the start state, equals h_score at the same time),
        # state itself, parent_state (=None since the start state has no parent, g_score) 
        
        # We create an iterator, which we will use later to distinguish open_list[], incase the current and next
        # state are the same and cannot be distinguished-
        counter = itertools.count()
        open_list =[(self.heuristic(start, goal), next(counter), start, None, 0)]

        # Closed list to track the states that have been visited.
        closed_list = set()

        # Dictionary to reconsruct the path later.
        came_from ={}

        ''' Main Search Loop'''

        # Begins a loop that continues as long as there are states to explore in
        # the open list.
        # We use heapq.heappop(open_list) to pop the state with the lowest f score.
        # The values are unpacked into '_' (ignored h-score), '_' (ignored counter) 'current_state (=start)', 'parent state', 
        # and g_score. 
        while open_list:
            _, _, current_state, parent_state, g_score = heapq.heappop(open_list)


            '''Checking and Updating the States'''

            # if the 'current state' has already been evaluated, we skip it.
            if current_state in closed_list:
                continue

            # Maps the 'current_state' to its 'parent_state' to reconstruct the path later.
            came_from[current_state] = parent_state


            '''Goal Check'''

            # If 'current_state' is the goal, reconstruct the path.
            # The path is then reversed to start from the initial state and returned by tracking
            # parent_state(s).
            if self.environment.is_goal(current_state):
                path = []


                
                while current_state:
                    path.append(current_state)
                    current_state = came_from[current_state]
                
                # The list 'path' is reversed.
                path.reverse()

                # 'environment attribute of the 'Agent'.
                # 'path' is a list of 'State' objects and rendered directly here.
                # 'render' method has functionality to stop: wait = True.
                self.environment.render(path=path, wait=True)
                return path

            # We add current_state to closed_list, so we skip it.
            closed_list.add(current_state)


            '''Till now, we have searched for possible paths.'''
        
            # Now, we evaluate the cost of actions for those selected paths.
            # For every action defined in environment, we calculate the cost. And thus, g_score.
            for action in self.environment.get_actions(current_state):
                new_state, cost = self.environment.get_transition_result(current_state, action)
                
                # If new_state included in the path, continue.
                if new_state in closed_list:
                    continue
                
                new_g = g_score + cost
                new_f = new_g + self.heuristic(new_state, goal)

                
                # We use heap queue algorithm, again. 
                # The first argument is open_list.
                # The algorithm ensures that 'open_list' remains a min-heap based on the 'f' score.
                # Thus A* algorithm efficinently selects the next node the explore, while next(counter) allows the equals states 
                # to be considered the same state without further confusion.
                # heapq.heappush, thus, generates the new state and sets current_state to parent_state and
                # restarts the loop.
                heapq.heappush(open_list, (new_f, next(counter), new_state, current_state, new_g))
                

        # If no path was found, return an empyt list.
        return None