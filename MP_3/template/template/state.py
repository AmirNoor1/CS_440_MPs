from abc import ABC, abstractmethod
import copy # NOTE: you may want to use copy when creating neighbors for EightPuzzle...
from typing import List


# NOTE: using this global index means that if we solve multiple 
#       searches consecutively the index doesn't reset to 0...
from itertools import count
global_index = count()

# TODO(III, IV): You should read through this abstract class.
#                Your search implementation must work with this API.
#                Specifically, your search will need to call is_goal() and generate_successors().
class SearchState(ABC):
    """Abstract base class for all search states"""
    def __init__(self, current_state, target_state, path_cost=0, enable_heuristic=True):
        self.current_state = current_state
        self.target_state = target_state
        # Unique identifier for tie-breaking in priority queue
        self.creation_order = next(global_index)
        # g(n) - actual cost from start to current state
        self.path_cost = path_cost
        self.enable_heuristic = enable_heuristic
        if enable_heuristic:
            self.heuristic_value = self.calculate_heuristic()
        else:
            self.heuristic_value = 0

    # Generate all valid successor states from current state
    @abstractmethod
    def generate_successors(self):
        pass
    
    # Check if current state satisfies goal condition
    @abstractmethod
    def goal_test(self):
        pass
    
    # Calculate heuristic estimate h(n) from current state to goal
    @abstractmethod
    def calculate_heuristic(self):
        pass

    # The "less than" method ensures that states are comparable
    #   meaning we can place them in a priority queue
    # You should compare states based on f = g + h = self.path_cost + self.h
    # Return True if self is less than other
    @abstractmethod
    def __lt__(self, other):
        # NOTE: if the two states (self and other) have the same f value, tiebreak using creation_order as below
        # so that the state created later is considered "less than" the one created earlier
        if self.creation_order > other.creation_order:
            return True

    # Hash function for visited state tracking
    @abstractmethod
    def __hash__(self):
        pass
    
    # Equality check for state comparison
    @abstractmethod
    def __eq__(self, other_state):
        pass
    
# LightsOut ------------------------------------------------------------------------------------------------

# State: List[List[{0, 1}]] of size m * n
# Goal: List[List[{0}]] of size m * n (all lights turn off)
class LightsOutState(SearchState):
    def __init__(self, state, goal, path_cost, enable_heuristic, cross_pattern=False):
        super().__init__(state, goal, path_cost, enable_heuristic)
        self.cross_pattern = cross_pattern

    # Helper function to check if a 2d index is a valid location in puzzle grid
    def _in_bounds(self, loc):
        if loc[0] < 0 or loc[0] > len(self.current_state)-1 or loc[1] < 0 or loc[1] > len(self.current_state[0])-1:
            return False
        return True
    
    # TODO(III): implement this method
    def generate_successors(self) -> List[SearchState]:
        nbr_states = []
        # NOTE: For reproducible tie-breaking, please add neighbors in row-major order.
        # For instance, you should toggle (0,0) first, then (0,1), ..., then (0,n-1), then (1,0), etc.
        return nbr_states

    def goal_test(self):
        # In python "==" performs deep list equality checking, so this works as desired
        return self.current_state == self.target_state
    
    # Convert state to tuples.
    def __hash__(self):
        return hash(self.__str__())
    def __eq__(self, other):
        return self.current_state == other.current_state

    # TODO(III): implement this method
    def calculate_heuristic(self):
        # The heuristic we use is the number of lights that are on divided by 5.
        pass

    # TODO(III): implement this method
    def __lt__(self, other):    
        # You should return True if the current state has a lower g + h value than "other"
        # If they have the same value then you should use creation_order to decide which is smaller
        pass
    
    # str and repr just make output more readable when your print out states
    def __str__(self):
        return str(self.current_state)
    def __repr__(self):
        return "\n".join([" ".join([str(c) for c in row]) for row in self.current_state])

# EightPuzzle ------------------------------------------------------------------------------------------------

# TODO(IV): implement this method (also need it for the next homework)
# Manhattan distance between two points (a=(a1,a2), b=(b1,b2))
def grid_distance(a, b):
    return 0

class EightPuzzleState(SearchState):
    def __init__(self, state, goal, path_cost, enable_heuristic, zero_loc):
        '''
        state: 3x3 array of integers 0-8
        goal: 3x3 goal array, default is np.arange(9).reshape(3,3).tolist()
        zero_loc: an additional helper argument indicating the 2d index of 0 in state, you do not have to use it
        '''
        # NOTE: SearchState constructor does not take zero_loc
        super().__init__(state, goal, path_cost, enable_heuristic)
        self.zero_loc = zero_loc
    
    # TODO(IV): implement this method
    def generate_successors(self):
        '''
        Return: a list of EightPuzzleState
        '''
        nbr_states = []
        # NOTE: There are *up to 4* possible neighbors and the order you add them matters for tiebreaking
        #   Please add them in the following order: [below, left, above, right], where for example "below" 
        #   corresponds to moving the empty tile down (moving the tile below the empty tile up)
        
        return nbr_states

    # Checks if goal has been reached
    def goal_test(self):
        # In python "==" performs deep list equality checking, so this works as desired
        return self.current_state == self.target_state
    
    # Can't hash a list, so first flatten the 2d array and then turn into tuple
    def __hash__(self):
        return hash(tuple([item for sublist in self.current_state for item in sublist]))
    def __eq__(self, other):
        return self.current_state == other.current_state
    
    # TODO(IV): implement this method
    def calculate_heuristic(self):
        total = 0
        # NOTE: There is more than one possible heuristic, 
        #       please implement the Manhattan heuristic, as described in the MP instructions
        
        return total
    
    # TODO(IV): implement this method
    # Hint: it should be identical to what you wrote in WordLadder.__lt__(self, other)
    def __lt__(self, other):
        pass
    
    # str and repr just make output more readable when you print out states
    def __str__(self):
        return str(self.current_state)
    def __repr__(self):
        return "\n".join([" ".join([str(c) for c in row]) for row in self.current_state])
    
