
"""
simpleSnake.py:
    this code is just for making a simple snake game, in which snake moves and 
    the user can get the randomly generated food (or target) to increas its score
    
    You can change the size of the display, the speed of the snake, waiting times
    for running, and some other parameters, explained in the text in front of 
    the lines.
    
    This version is too simple and you can increase more functionality easily. In
    the other two codes, uploaded in the same repository in github, you can find
    more difficult implementation of Snake game using AI and reinforcement learning.
    
    Please send a message if you have any question: mahmoud.ramezani@ui.no
"""

import pygame, random, sys
from pygame.locals import QUIT, KEYDOWN, K_UP, K_DOWN, K_LEFT, K_RIGHT


def collide(x1, x2, y1, y2, w1, w2, h1, h2):
    """This funciton detect any collision of snake with borders of the screen """
    if x1 + w1 > x2 and x1 < x2 + w2 and y1 + h1 > y2 and y1 < y2 + h2:
        return True
    else:
        return False


def die(screen, score):
    """Printing the final score and closing the window after a couple of seconds
    This function operates when the user loose the game. """
    f = pygame.font.SysFont('Arial', 30)
    t = f.render('You lost! Your score is: ' + str(score), True, (0, 0, 0))
    screen.blit(t, (10, 270))
    pygame.display.update()
    while t:
        for e in pygame.event.get():
            if e.type == QUIT:            # if the user wants to quit the game by closing the window
                pygame.quit() 
                sys.exit()
    return 1
    
    
xs = [290, 290, 290, 290, 290];
ys = [290, 270, 250, 230, 210];
dirs = 0;
score = 0;
targetpos = (random.randint(0, 590), random.randint(0, 590)); #random positions generation of the target
pygame.init();
s = pygame.display.set_mode((600, 600));
pygame.display.update();
pygame.display.set_caption('Snake');
targetimage = pygame.Surface((10, 10)); # the size of the target
targetimage.fill((0, 255, 0));
img = pygame.Surface((20, 20)); # the size of each squares constructing the snake
img.fill((255, 0, 0));
f = pygame.font.SysFont('Arial', 20);
clock = pygame.time.Clock()
pygame.time.wait(2000)
while True:
    clock.tick(10)                    #Adjusting the speed of the snake
    for e in pygame.event.get():
        if e.type == QUIT:            # if the user wants to quit the game by closing the window
            pygame.quit() 
            # sys.exit(0)
        elif e.type == KEYDOWN:      # Arrow key assignment to the moves
            if e.key == K_UP and dirs != 0:
                dirs = 2
            elif e.key == K_DOWN and dirs != 2:
                dirs = 0
            elif e.key == K_LEFT and dirs != 1:
                dirs = 3
            elif e.key == K_RIGHT and dirs != 3:
                dirs = 1
    i = len(xs) - 1
    while i >= 2:
        if collide(xs[0], xs[i], ys[0], ys[i], 20, 20, 20, 20): die(s, score)
        i -= 1
    if collide(xs[0], targetpos[0], ys[0], targetpos[1], 20, 10, 20, 10): score += 1;xs.append(700);ys.append(
        700);targetpos = (random.randint(0, 590), random.randint(0, 590))
    if xs[0] < 0 or xs[0] > 580 or ys[0] < 0 or ys[0] > 580: die(s, score)
    i = len(xs) - 1
    while i >= 1:
        xs[i] = xs[i - 1];
        ys[i] = ys[i - 1];
        i -= 1
    if dirs == 0:
        ys[0] += 20
    elif dirs == 1:
        xs[0] += 20
    elif dirs == 2:
        ys[0] -= 20
    elif dirs == 3:
        xs[0] -= 20
    s.fill((255, 255, 255))
    for i in range(0, len(xs)):
        s.blit(img, (xs[i], ys[i]))
    s.blit(targetimage, targetpos);
    t = f.render(str(score), True, (0, 0, 0));
    s.blit(t, (10, 10));
    pygame.display.update()





