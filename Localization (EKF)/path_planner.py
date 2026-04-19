import numpy as np
import heapq

class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0  # Path cost from start to current node
        self.h = 0  # Heuristic cost from current node to goal
        self.f = 0  # Total node cost (g + h)
    
    # Required for heapq to handle f-score ties
    def __lt__(self, other):
        return self.f < other.f

# --- CONVERSION FUNCTIONS ---
def world_to_grid(world_x, world_y, resolution=10, map_offset=5):
    grid_x = int(round((world_x + map_offset) * resolution))
    grid_y = int(round((world_y + map_offset) * resolution))
    return (grid_x, grid_y)

def grid_to_world(grid_x, grid_y, resolution=10, map_offset=5):
    world_x = (grid_x / resolution) - map_offset
    world_y = (grid_y / resolution) - map_offset
    return (world_x, world_y)

# --- A* PATHFINDING ALGORITHM ---
def A_star(grid, start, goal):
    # --- INITIAL SAFETY ADJUSTMENT ---
    # Clear the start node if it's on/near an obstacle to ensure the pathfinder starts successfully.
    grid[start[1]][start[0]] = 0 

    start_node = Node(start, None)
    goal_node = Node(goal, None)
    
    open_list = []
    heapq.heappush(open_list, (start_node.f, start_node))
    visited_costs = {start: 0}

    while open_list:
        _, current_node = heapq.heappop(open_list)

        if current_node.position == goal_node.position:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]

        # Explore 8 neighbors
        for new_pos in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            node_pos = (current_node.position[0] + new_pos[0], 
                        current_node.position[1] + new_pos[1])

            # Check boundaries
            if (node_pos[0] >= grid.shape[1] or node_pos[0] < 0 or 
                node_pos[1] >= grid.shape[0] or node_pos[1] < 0):
                continue
            
            # Skip if cell is an obstacle
            if grid[node_pos[1]][node_pos[0]] == 1:
                continue
                
            # --- POTENTIAL FIELD (Safety cost for obstacle proximity) ---
            safety_cost = 0
            if np.any(grid[max(0, node_pos[1]-2):node_pos[1]+3, 
                           max(0, node_pos[0]-2):node_pos[0]+3] == 1):
                safety_cost = 2.0 
            
            move_cost = 1.414 if abs(new_pos[0]) == 1 and abs(new_pos[1]) == 1 else 1.0
            new_g = current_node.g + move_cost + safety_cost
            
            if node_pos in visited_costs and new_g >= visited_costs[node_pos]:
                continue
            
            visited_costs[node_pos] = new_g
            child = Node(node_pos, current_node)
            child.g = new_g
            
            # Pure Euclidean Heuristic
            child.h = np.sqrt((node_pos[0] - goal_node.position[0])**2 + 
                              (node_pos[1] - goal_node.position[1])**2) 
            child.f = child.g + child.h
            
            heapq.heappush(open_list, (child.f, child))
            
    return None

# --- PATH SMOOTHING ---
def has_line_of_sight(p1, p2, grid):
    # Rigorous check to prevent cutting corners
    steps = int(max(abs(p2[0]-p1[0]), abs(p2[1]-p1[1]))*3)
    if steps == 0: return True
    for i in range(steps + 1):
        t = i / steps
        curr_x = int(round(p1[0] + (p2[0] - p1[0]) * t))
        curr_y = int(round(p1[1] + (p2[1] - p1[1]) * t))
        # Verify if it hits an obstacle or safety buffer zone
        if grid[curr_y][curr_x] == 1:
            return False
    return True

def smooth_path(path, grid):
    if not path or len(path) <= 2: return path
    smoothed = [path[0]]
    curr = 0
    while curr < len(path) - 1:
        best_next = curr + 1
        # Look-ahead reduced to 6 for smoother curves in 10x10m arena
        look_ahead = min(curr + 6, len(path) - 1)
        for next_idx in range(look_ahead, curr + 1, -1):
            if has_line_of_sight(path[curr], path[next_idx], grid):
                best_next = next_idx
                break
        smoothed.append(path[best_next])
        curr = best_next
    return smoothed
