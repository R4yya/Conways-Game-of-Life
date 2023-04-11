import pygame
from pygame.locals import *
from random import randint, choice
from copy import deepcopy
import numpy as np
from numba import njit


class GameOfLife():
    _RESOLUTION = _WIDTH, _HEIGHT = 1600, 900
    _CELL_SIZE = 15
    _CELL_X_AMOUNT = _WIDTH // _CELL_SIZE
    _CELL_Y_AMOUNT = _HEIGHT // _CELL_SIZE
    _FPS_MAX = 24
    _PAUSED = False
    _BG_COLOR = (5, 5, 5)
    _LINES_COLOR = (15, 15, 15)
    _CELL_PALETES = {'red': [(70, 5, 10),
                             (90, 10, 20),
                             (110, 15, 30)],

                     'green': [(5, 60, 15),
                               (10, 90, 30),
                               (15, 120, 45)],

                     'blue': [(0, 30, 60),
                              (0, 60, 120),
                              (0, 80, 160)],

                     'orange': [(239, 84, 17),
                                (255, 101, 23),
                                (255, 130, 46)],

                     'yellow': [(255, 232, 20),
                                (255, 177, 20),
                                (255, 196, 18)],

                     'purple': [(37, 6, 76),
                                (54, 23, 94),
                                (85, 50, 133)],

                     'white': [(166, 166, 166),
                               (198, 198, 198),
                               (224, 224, 224)]}

    def __init__(self):
        self.screen = pygame.display.set_mode(self._RESOLUTION)
        self.clock = pygame.time.Clock()

        self.create_grids()

        self.selected_pallete = 'white'

        pygame.init()
        pygame.display.set_caption('Game Of Life')

    def create_grids(self):
        self.next_grid = np.array(
            [[0 for i in range(self._CELL_X_AMOUNT)]
             for j in range(self._CELL_Y_AMOUNT)])

        self.current_grid = np.array(
            [[randint(0, 1) for i in range(self._CELL_X_AMOUNT)]
             for j in range(self._CELL_Y_AMOUNT)])

    def clear_grids(self):
        self.next_grid = np.array(
            [[0 for i in range(self._CELL_X_AMOUNT)]
             for j in range(self._CELL_Y_AMOUNT)])

        self.current_grid = np.array(
            [[0 for i in range(self._CELL_X_AMOUNT)]
             for j in range(self._CELL_Y_AMOUNT)])

    def get_selected_cell_position(self):
        self.mouse_pos_x, self.mouse_pos_y = pygame.mouse.get_pos()
        self.mouse_pos_x //= self._CELL_SIZE
        self.mouse_pos_y //= self._CELL_SIZE

        return self.mouse_pos_x, self.mouse_pos_y

    def revive_selected_cell(self):
        self.cell_pos_x, self.cell_pos_y = self.get_selected_cell_position()

        if (self.cell_pos_x, self.cell_pos_y) not in self.cells_to_draw:
            self.cells_to_draw.append((self.cell_pos_x, self.cell_pos_y))
            try:
                self.current_grid[self.cell_pos_y][self.cell_pos_x] = 1
            except IndexError:
                pass

        self.screen.fill(pygame.Color(self._BG_COLOR))
        self.draw_lines()
        self.draw_cells()

    def kill_selected_cell(self):
        self.cell_pos_x, self.cell_pos_y = self.get_selected_cell_position()

        try:
            self.cells_to_draw.remove((self.cell_pos_x, self.cell_pos_y))
            self.current_grid[self.cell_pos_y][self.cell_pos_x] = 0
        except ValueError:
            pass
        except IndexError:
            pass

        self.screen.fill(pygame.Color(self._BG_COLOR))
        self.draw_lines()
        self.draw_cells()

    def choose_colors(self, key):
        return choice(self._CELL_PALETES.get(key))

    def draw_lines(self):
        for x in range(0, self._WIDTH, self._CELL_SIZE):
            pygame.draw.line(self.screen, pygame.Color(self._LINES_COLOR),
                             (x, self._CELL_SIZE),
                             (x, self._HEIGHT - self._CELL_SIZE))

        for y in range(0, self._HEIGHT, self._CELL_SIZE):
            pygame.draw.line(self.screen, pygame.Color(self._LINES_COLOR),
                             (self._CELL_SIZE, y),
                             (self._WIDTH - self._CELL_SIZE, y))

    def draw_cells(self):
        [pygame.draw.rect(self.screen,
                          pygame.Color(self.choose_colors(
                              self.selected_pallete)),
                          (x * self._CELL_SIZE + 1,
                           y * self._CELL_SIZE + 1,
                           self._CELL_SIZE - 1,
                           self._CELL_SIZE - 1))for x, y in self.cells_to_draw]

    @staticmethod
    @njit(fastmath=True)
    def check_neighbours(current_grid,
                         next_grid,
                         _CELL_X_AMOUNT,
                         _CELL_Y_AMOUNT):
        cells_to_draw = []

        for x in range(1, _CELL_X_AMOUNT - 1):
            for y in range(1, _CELL_Y_AMOUNT - 1):

                count = 0

                for j in range(y - 1, y + 2):
                    for i in range(x - 1, x + 2):
                        if current_grid[j][i] == 1:
                            count += 1

                if current_grid[y][x] == 1:
                    count -= 1
                    if count == 2 or count == 3:
                        next_grid[y][x] = 1
                        cells_to_draw.append((x, y))
                    else:
                        next_grid[y][x] = 0
                else:
                    if count == 3:
                        next_grid[y][x] = 1
                        cells_to_draw.append((x, y))
                    else:
                        next_grid[y][x] = 0

        return next_grid, cells_to_draw

    def use_njit_function(self):
        current_grid = self.current_grid
        next_grid = self.next_grid
        _CELL_X_AMOUNT = self._CELL_X_AMOUNT
        _CELL_Y_AMOUNT = self._CELL_Y_AMOUNT
        return self.check_neighbours(current_grid,
                                     next_grid,
                                     _CELL_X_AMOUNT,
                                     _CELL_Y_AMOUNT)

    def get_next_generation(self):
        self.next_grid, self.cells_to_draw = self.use_njit_function()
        self.draw_cells()
        self.current_grid = deepcopy(self.next_grid)

    def kill_generation(self):
        self.screen.fill(pygame.Color(self._BG_COLOR))
        self.draw_lines()
        self.clear_grids()
        self.get_next_generation()

    def do_one_step(self):
        self.screen.fill(pygame.Color(self._BG_COLOR))
        self.draw_lines()
        self.get_next_generation()

    def create_new_generation(self):
        self.screen.fill(pygame.Color(self._BG_COLOR))
        self.draw_lines()
        self.create_grids()
        self.get_next_generation()

    def run_game(self):
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    exit()
                elif event.type == KEYDOWN:
                    if event.key == K_1:
                        self.selected_pallete = 'red'
                        self.draw_cells()

                    elif event.key == K_2:
                        self.selected_pallete = 'green'
                        self.draw_cells()

                    elif event.key == K_3:
                        self.selected_pallete = 'blue'
                        self.draw_cells()

                    elif event.key == K_4:
                        self.selected_pallete = 'orange'
                        self.draw_cells()

                    elif event.key == K_5:
                        self.selected_pallete = 'yellow'
                        self.draw_cells()

                    elif event.key == K_6:
                        self.selected_pallete = 'purple'
                        self.draw_cells()

                    elif event.key == K_SPACE:
                        self._PAUSED = not self._PAUSED

                    elif event.key == K_s:
                        self.do_one_step()
                        print(1)

                    elif event.key == K_r:
                        self.create_new_generation()

                    elif event.key == K_c:
                        self.kill_generation()

            clicks = pygame.mouse.get_pressed()
            if clicks[0]:
                self.revive_selected_cell()
            elif clicks[2]:
                self.kill_selected_cell()

            if not self._PAUSED:
                self.screen.fill(pygame.Color(self._BG_COLOR))
                pygame.display.set_caption(
                    'Game Of Life ' + str(int(self.clock.get_fps())) + ' FPS')
                self.draw_lines()
                self.get_next_generation()

            pygame.display.flip()
            self.clock.tick(self._FPS_MAX)


if __name__ == '__main__':
    game = GameOfLife()
    game.run_game()
