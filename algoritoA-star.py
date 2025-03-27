import pygame
import math
from queue import PriorityQueue
from typing import List, Tuple, Optional

# Constants
WIDTH = 800  # Increased window size for better visibility
ROWS = 50
COLORS = {
    'WHITE': (255, 255, 255),
    'BLACK': (0, 0, 0),
    'GREY': (128, 128, 128),
    'BLUE': (0, 0, 255),
    'ORANGE': (255, 165, 0),
    'GREEN': (0, 255, 0),
    'PURPLE': (128, 0, 128),
    'RED': (255, 0, 0),
    'TURQUOISE': (64, 224, 208)
}

class Node:
    def __init__(self, row: int, col: int, width: int, total_rows: int):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = COLORS['WHITE']
        self.neighbors = []
        self.width = width
        self.total_rows = total_rows

    def get_pos(self) -> Tuple[int, int]:
        return self.row, self.col

    def is_closed(self) -> bool:
        return self.color == COLORS['RED']

    def is_open(self) -> bool:
        return self.color == COLORS['GREEN']

    def is_barrier(self) -> bool:
        return self.color == COLORS['BLACK']

    def is_start(self) -> bool:
        return self.color == COLORS['ORANGE']

    def is_end(self) -> bool:
        return self.color == COLORS['TURQUOISE']

    def reset(self):
        self.color = COLORS['WHITE']

    def make_start(self):
        self.color = COLORS['ORANGE']

    def make_closed(self):
        self.color = COLORS['RED']

    def make_open(self):
        self.color = COLORS['GREEN']

    def make_barrier(self):
        self.color = COLORS['BLACK']

    def make_end(self):
        self.color = COLORS['TURQUOISE']

    def make_path(self):
        self.color = COLORS['PURPLE']

    def draw(self, win: pygame.Surface):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

    def update_neighbors(self, grid: List[List['Node']]):
        self.neighbors = []
        # Down
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier():
            self.neighbors.append(grid[self.row + 1][self.col])
        # Up
        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier():
            self.neighbors.append(grid[self.row - 1][self.col])
        # Right
        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier():
            self.neighbors.append(grid[self.row][self.col + 1])
        # Left
        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier():
            self.neighbors.append(grid[self.row][self.col - 1])

    def __lt__(self, other):
        return False


def h(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """Heuristic function (Manhattan distance)"""
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)


def reconstruct_path(came_from: dict, current: Node, draw_func):
    """Reconstruct and draw the path from end to start"""
    while current in came_from:
        current = came_from[current]
        current.make_path()
        draw_func()


def a_star_algorithm(draw_func, grid: List[List[Node]], start: Node, end: Node) -> bool:
    """A* pathfinding algorithm implementation"""
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    
    # g_score: cost from start to current node
    g_score = {node: float("inf") for row in grid for node in row}
    g_score[start] = 0
    
    # f_score: estimated total cost from start to end through current node
    f_score = {node: float("inf") for row in grid for node in row}
    f_score[start] = h(start.get_pos(), end.get_pos())
    
    open_set_hash = {start}
    
    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
        
        current = open_set.get()[2]
        open_set_hash.remove(current)
        
        if current == end:
            reconstruct_path(came_from, end, draw_func)
            end.make_end()
            return True
        
        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1
            
            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos())
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()
        
        draw_func()
        
        if current != start:
            current.make_closed()
    
    return False


def make_grid(rows: int, width: int) -> List[List[Node]]:
    """Create a grid of nodes"""
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            node = Node(i, j, gap, rows)
            grid[i].append(node)
    return grid


def draw_grid(win: pygame.Surface, rows: int, width: int):
    """Draw grid lines"""
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, COLORS['GREY'], (0, i * gap), (width, i * gap))
        pygame.draw.line(win, COLORS['GREY'], (i * gap, 0), (i * gap, width))


def draw(win: pygame.Surface, grid: List[List[Node]], rows: int, width: int):
    """Draw all elements"""
    win.fill(COLORS['WHITE'])
    
    for row in grid:
        for node in row:
            node.draw(win)
    
    draw_grid(win, rows, width)
    pygame.display.update()


def get_clicked_pos(pos: Tuple[int, int], rows: int, width: int) -> Tuple[int, int]:
    """Get grid position from mouse coordinates"""
    gap = width // rows
    y, x = pos
    row = y // gap
    col = x // gap
    return row, col


def main(win: pygame.Surface, width: int):
    """Main game loop"""
    grid = make_grid(ROWS, width)
    
    start: Optional[Node] = None
    end: Optional[Node] = None
    
    run = True
    started = False
    
    while run:
        draw(win, grid, ROWS, width)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            
            if started:
                continue
            
            # Left mouse button - place nodes
            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                node = grid[row][col]
                
                if not start and node != end:
                    start = node
                    start.make_start()
                elif not end and node != start:
                    end = node
                    end.make_end()
                elif node != end and node != start:
                    node.make_barrier()
            
            # Right mouse button - reset nodes
            elif pygame.mouse.get_pressed()[2]:
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                node = grid[row][col]
                node.reset()
                if node == start:
                    start = None
                elif node == end:
                    end = None
            
            # Start algorithm with SPACE
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start and end:
                    for row in grid:
                        for node in row:
                            node.update_neighbors(grid)
                    
                    a_star_algorithm(lambda: draw(win, grid, ROWS, width), grid, start, end)
                
                # Reset with 'c'
                if event.key == pygame.K_c:
                    start = None
                    end = None
                    grid = make_grid(ROWS, width)
    
    pygame.quit()


if __name__ == "__main__":
    pygame.init()
    WIN = pygame.display.set_mode((WIDTH, WIDTH))
    pygame.display.set_caption("A* Pathfinding Algorithm")
    main(WIN, WIDTH)