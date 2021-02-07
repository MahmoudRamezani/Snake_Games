

"""
Snake_AI.py: (please first read the code simpleSnake.py at the same repository)
    this code runs the snake game by reinforcement learning. The AI user tries
    to find the optimum path to the target (food for the snake!) by applying
    a heuristic function. Three categories of functions have been written and 
    you can choose as you wish. The first one called "find_path_bfs" is the 
    well-known breadth-first-search algorithm, the seconed one "find_path_dfs" 
    is the depth-first-search algorithm and the last one is "find_path_A" is the 
    one that the user can write it. In the user defined function, as of now, 
    four heuristic functions are defined here and you can
    choose one of them. They are generally based on the Manhattan and root mean 
    square concepts.Also, you can define your desired heuristic function.
    
    There is a blocked square in the screen that the snake can not go through it
    and should bypass it (we show the score of the game exactly in that square
    by a number). 
    To learn more about the search functions in reinforcement learning, please look
    at the following reference which also used here
    https://www.tutorialspoint.com/artificial_intelligence/artificial_intelligence_popular_search_algorithms.htm

    please contact mahmoud.ramezani@uia.no if you have any question.
"""
import pygame, random, sys
import numpy
import heapq
import collections
import pdb
import time
from collections import deque

s_pixel=20 #This is the lenght in pixel for an square in the screen
numrow=30
numcol=30

DIRECTIONS = [(0, -1, 'U'), (0, 1, 'D'), (-1, 0, 'L'), (1, 0, 'R')] # All possible moves, up, down, left, right
Node = collections.namedtuple('Node', ['hist', 'ix', 'dist', 'pt', 'prev', 'direction'])
# hist: the heuristic measure for the node to the goal
# ix:   a serial number to ensure that subsequent fields are not compared.
# dist: the distance of the node to the goal
# pt:   refers to the point (x,y), current coordinates
# Prev: refers to the previous node, because we want to come back to the source

blockpos = random.randint(0, numcol-1), random.randint(0, numrow-1)
block_pos=numpy.array(blockpos)
blockpos=block_pos*s_pixel

# The followings are different heuristic functions used for the distance optimization problem
def heuristic1(point, goal):
    """Manhattan distance"""
    return abs(point[0] - goal[0]) + abs(point[1] - goal[1])


def heuristic2 (point, goal):
    """root mean square (RMS)"""
    return ((point[0] - goal[0])** 2 + (point[1] - goal[1])**2)**.5 


def heuristic3 (point, goal):
    """the average of RMS and Manhattan"""
    return (heuristic1(point, goal)+heuristic2(point, goal))/2


def heuristic4 (point, goal):
    """A function of Manhattan, you can insert your desired function here and use it!"""
    return 3*abs(point[0] - goal[0]) + abs(point[1] - goal[1]) 


def get_adjacent(coords, x_blk, y_blk, x_snake, y_snake):
    x, y = coords
    coords = [(x - 1, y, "L"), (x, y - 1, "U"),
              (x + 1, y, "R"), (x, y + 1, "D")]
    adjSquares = [((x, y), c) for x, y, c in coords if
               x > -1 and x < numcol and y > -1 and y < numrow and (x, y) != (x_blk, y_blk) and
               (x, y) != (x_snake[1], y_snake[1]) and (x, y) != (x_snake[2], y_snake[2])]
    return adjSquares


def find_path_bfs (x_blk_pos , y_blk_pos, x_snake, y_snake, goal):
    """Breadth First search algorithm:
    It starts from the root node, explores the neighboring nodes first and 
    moves towards the next level neighbors. It generates one tree at a time 
    until the solution is found. It can be implemented using FIFO queue data 
    structure. This method provides shortest path to the solution.
    A lot of memory space is required!
    """
    start=(x_snake[0], y_snake[0])
    q= deque([(start, "")])
    v=set()
    while q:
        cords, path = q.popleft()
        if cords == goal:
            return path
        if cords in v:
            continue
        v.add(cords)
        for pos, mark in get_adjacent(cords, x_blk_pos , y_blk_pos, x_snake, y_snake):
            if pos in v:
                continue
            else:
                q.append((pos, path + mark))


def find_path_dfs (x_blk_pos , y_blk_pos, x_snake, y_snake, goal):
    """
    Depth First search algorithm:
        It is implemented in recursion with LIFO stack data structure. It creates
        the same set of nodes as Breadth-First method, only in the different order.
        As the nodes on the single path are stored in each iteration from root to 
        leaf node, the space requirement to store nodes is linear. 
    
    """
    start=(x_snake[0], y_snake[0])
    q= deque([(start, "")])
    v=set()
    while q:
        cords, path = q.pop()
        if cords == goal:
            return path
        if cords in v:
            continue
        v.add(cords)
        for pos, mark in get_adjacent(cords, x_blk_pos , y_blk_pos, x_snake, y_snake):
            if pos in v:
                continue
            else:
                q.append((pos, path + mark))


def find_path_A (x_blk_pos , y_blk_pos, x_snake, y_snake, goal):
    init_matrix = numpy.zeros((numcol, numrow))
    init_matrix[int(x_blk_pos)][int(y_blk_pos)]=1
    init_matrix[int(x_snake[1])][int(y_snake[1])] = 1
    init_matrix[int(x_snake[3])][int(y_snake[3])] = 1
    start=(x_snake[0],y_snake[0])
    # Each node consists of (estimated path distance, ix, dist, (x, y), previous node, direction)
    open = [Node(heuristic4(start, goal), 0, 0, start, None, None)]
    explored = set()   # A set of all visited coordinates.
    ix = 1
    while open:
        node = heapq.heappop(open)
        _, _, dist, point, prev, prev_d = node
        if point in explored:
            continue
        if point == goal:
            break
        explored.add(point)
        # Now consider moves in each direction.
        for dx, dy, d in DIRECTIONS:
            new_point = point[0] + dx, point[1] + dy
            if new_point not in explored \
                    and new_point[0] > -1 and new_point[0] < numcol and new_point[1] > -1 and new_point[1] < numrow\
                    and not init_matrix[int(new_point[0])][int(new_point[1])] \
                    and new_point!=(x_snake[1], y_snake[1]) and new_point!=(x_snake[2], y_snake[2]):
                h = dist + 1 + heuristic1(new_point, goal)
                tie_break = 4 if prev_d != d else 0  # Prefer moving straight
                new_node = Node(h, ix + tie_break, dist + 1, new_point, node, d)
                heapq.heappush(open, new_node)
                ix = ix + 1
    # Return a path to node
    result = ''
    while node.prev is not None:
        result = node.direction + result
        node = node.prev
    return result


xs = [180, 180, 180, 180, 180]
ys = [180, 160, 140, 120, 100]
score = 0
target_pos = (random.randint(0, numcol-1), random.randint(0, numrow-1));
target_pos = numpy.array(target_pos)
targetpos = target_pos*s_pixel

pygame.init()
s = pygame.display.set_mode((numcol*s_pixel, numrow*s_pixel))     # design the main screen
pygame.display.set_caption('Snake by AI')
targetimage = pygame.Surface((s_pixel, s_pixel))       # creating a target with size of (10,10)!
targetimage.fill((0, 0, 0))               # color of target
img = pygame.Surface((s_pixel, s_pixel))              # creating the squares and size of the squares of snake
img.fill((128, 0, 128))                     # color of snake body
imp_block=pygame.Surface((s_pixel, s_pixel))          # impassable block
imp_block.fill((255, 0, 0))
f = pygame.font.SysFont('Arial', 20);        # font size of the score on the screen
clock = pygame.time.Clock()                  # create an object to help track time
while True:
    xspos = numpy.array(xs)
    xs_pos=xspos/s_pixel
    yspos = numpy.array(ys)
    ys_pos = yspos / s_pixel
    # Here, you have to choose which heuristic function you want as the search algorithm
    chosen_path = find_path_A (block_pos[0], block_pos[1], xs_pos, ys_pos, (target_pos[0],target_pos[1]))
    print (chosen_path)
    CounterToExit=10
    for step_move in range (0, len(chosen_path)):
        for e in pygame.event.get():  # receive input from the keyboard
            if e.type == pygame.QUIT:  # if user close the window when game is running
                pygame.quit()
                sys.exit()
            elif e.type == pygame.KEYDOWN: #This is just to have an interupt if the user wants
                time.sleep(3)
                continue

        clock.tick(20)    #adjust the speed of running
        if step_move==len(chosen_path)-1:# if snake's head collides with target
            score += 1;
            target_pos=(random.randint(0, numcol - 1), random.randint(0, numrow - 1))
            target_pos = numpy.array(target_pos)
            targetpos = target_pos * s_pixel

        dirs = chosen_path[step_move]
        i = len(xs) - 1
        while i >= 1:
            xs[i] = xs[i - 1];
            ys[i] = ys[i - 1];
            i -= 1
        if dirs == 'D':                             # DOWN
            ys[0] += s_pixel
        elif dirs == 'R':                           # RIGHT
            xs[0] += s_pixel
        elif dirs == 'U':                           # UP
            ys[0] -= s_pixel
        elif dirs == 'L':                           # LEFT
            xs[0] -= s_pixel
        s.fill((255, 255, 255))                   # screen background color= white
        for i in range(1, len(xs)-1):
            s.blit(img, (xs[i], ys[i]))           # .blit is used for draw one image onto another

        img_head = pygame.Surface((s_pixel, s_pixel));
        img_head.fill((0, 255, 0));               # head should be Green
        s.blit(img_head, (xs[0], ys[0]))

        img_tail = pygame.Surface((s_pixel, s_pixel));
        img_tail.fill((0, 0, 255));               # tail should be blue
        s.blit(img_tail, (xs[len(xs)-1], ys[len(xs)-1]))

        s.blit(imp_block, blockpos);         # placing impassable block

        s.blit(targetimage, targetpos);
        t = f.render(str(score), True, (0, 0, 0));
        s.blit(t, (blockpos[0], blockpos[1]));                      # the cordinates of score's place

        pygame.display.update()