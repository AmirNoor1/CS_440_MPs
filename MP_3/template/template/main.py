from state import LightsOutState, EightPuzzleState
from search import astar_search
from utils import read_eight_puzzle, read_lights_out, get_goal_lights_out

import time
import argparse

def main(args):
    if args.problem_type == "LightsOut":
        lights_out_problems = read_lights_out(args.lights_out_file)
        for grid, cross_pattern, ground_truth_len in lights_out_problems:
            print("-"*40)
            print(f"Doing LightsOut {'(cross pattern)' if cross_pattern else '(original)'} for grid:")
            print("\n".join([" ".join([str(c) for c in row]) for row in grid]))
            goal = get_goal_lights_out(grid)
            start = time.time()
            starting_state = LightsOutState(grid, goal=goal, 
                                path_cost=0, enable_heuristic=not args.do_not_use_heuristic,
                                cross_pattern=cross_pattern)
            path = astar_search(starting_state)
            end = time.time()
            if args.print_solution:
                print("Solution:")
                for i, state in enumerate(path):
                    print(f"  (State {i})")
                    print(state.__repr__())
            print("\tNumber of toggles (ground truth)  : ", ground_truth_len)
            print("\tNumber of toggles in your solution: ", len(path) - 1)
            print(f"\tTime: {end-start:.3f}")

    elif args.problem_type == "EightPuzzle":
        print(f"Doing EightPuzzle for length {args.puzzle_len} puzzles")
        all_puzzles = read_eight_puzzle(f"data/eight_puzzle/{args.puzzle_len}_moves.txt")
        for puzzle in all_puzzles:
            print("-"*40)
            start = time.time()
            start_puzzle = puzzle[0]
            zero_loc = puzzle[1]
            print(f"Start puzzle: {start_puzzle}")
            goal_puzzle = [[0,1,2],[3,4,5],[6,7,8]]
            starting_state = EightPuzzleState(start_puzzle, goal_puzzle, 
                                path_cost=0, enable_heuristic=not args.do_not_use_heuristic, zero_loc=zero_loc)
            path = astar_search(starting_state)
            end = time.time()
            print("Solution:")
            for i, state in enumerate(path):
                print(f"  (State {i})")
                print(state.__repr__())
            print("\tNumber of moves (ground truth)  : ", args.puzzle_len)
            print("\tNumber of moves in your solution: ", len(path) - 1)
            print(f"\tTime: {end-start:.3f}")

    else:
        print("Problem type must be one of [LightsOut, EightPuzzle]")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CS440 MP3 Search')
    parser.add_argument('--problem_type',dest="problem_type", type=str,default="EightPuzzle",
                        help='Which search problem (i.e., State) to solve: [LightsOut, EightPuzzle]')
    parser.add_argument('--do_not_use_heuristic', action = 'store_true',
                        help = 'Do not use heuristic h in astar_search')
    parser.add_argument('--print_solution', action = 'store_true',
                        help = 'Print out the full solution path for debugging')

    # LIGHTSOUT ARGS
    parser.add_argument('--lights_out_file', dest="lights_out_file", type=str, default = "data/lights_out/open.txt",
                        help='File containing LightsOut problems')

    # EIGHTPUZZLE ARGS
    parser.add_argument('--eight_puzzle_len',dest="puzzle_len", type=int, default = 5,
                        help='EightPuzzle problem difficulty: one of [5, 10, 27]')

    args = parser.parse_args()
    main(args)