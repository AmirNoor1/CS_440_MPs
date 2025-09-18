
# EightPuzzle ------------------------------------------------------------------------------------------------

def read_eight_puzzle(filename):
    with open(filename, "r") as file:
        all_grids = []
        for line in file:
            grid = [[]]
            for c in line.strip():
                if len(grid[-1])==3:
                    grid.append([])
                intc = int(c)
                if intc == 0:
                    zero_loc = [len(grid)-1, len(grid[-1])]
                grid[-1].append(intc)
            all_grids.append([grid, zero_loc])
        return all_grids

# LightsOut ------------------------------------------------------------------------------------------------
def read_lights_out(filename):
    with open(filename, "r") as file:
        all_grids = []

        current_grid_size = None
        for line in file:
            # Line for puzzle size info example: "# 4 4 +" / "# 3 3 X"
            if line.startswith("#"):
                current_grid_size = int(line.split(" ")[1]), int(line.split(" ")[2])
                current_grid_pattern = line.split(" ")[3].strip()
                assert current_grid_pattern in ["+", "X"]
                ground_truth_len = int(line.split(" ")[4].strip())
            else:
                assert len(line.strip()) == current_grid_size[0]*current_grid_size[1]
                grid = []
                for r in range(current_grid_size[0]):
                    row = []
                    for c in range(current_grid_size[1]):
                        char = line[r*current_grid_size[1]+c]
                        if char == "0":
                            row.append(0)
                        elif char == "1":
                            row.append(1)
                        else:
                            raise ValueError(f"Invalid character {char} in LightsOut puzzle file {filename}")
                    grid.append(row)
                all_grids.append((grid, current_grid_pattern=="X", ground_truth_len))
        return all_grids

def get_goal_lights_out(state):
    return [[0 for _ in range(len(state[0]))] for _ in range(len(state))]