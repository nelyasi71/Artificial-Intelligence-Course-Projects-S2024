from cube import Cube
from constants import *
from utility import *

import random
import numpy as np

class Snake:
    body = []
    turns = {}

    def __init__(self, color, pos, file_name=None):
        self.color = color
        self.head = Cube(pos, color=color)
        self.body.append(self.head)
        self.dirnx = 0
        self.dirny = 1
        self.last_direction = (self.dirnx, self.dirny)
        try:
            self.q_table = np.load('qtable.npy')
        except:
            self.q_table = np.zeros(shape = (11,4,2,2,2,2,2,2,2,2,2,2,4))             
        
        self.lr = 0.05 
        self.discount_factor = 1 
        self.epsilon = 0.05
        # self.min_epsilon = 0.05 
        # self.epsilon_decay = 0.95
        self.num_episodes = 100
        self.action_counter = 0 
        
    def get_optimal_policy(self, state):
        return np.argmax(self.q_table[state])

    def make_action(self, state):
        chance = random.random()
        if chance < self.epsilon:
            action = random.randint(0, 3)
        else:
            action = self.get_optimal_policy(state)

        # self.action_counter += 1
        # if self.action_counter == self.num_episodes:
        #     self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        #     print(self.epsilon)
        #     self.action_counter = 0
        return action

    def update_q_table(self, state, action, next_state, reward):
        self.q_table[state][action] = self.q_table[state][action] + self.lr * (reward + self.discount_factor * max(self.q_table[next_state]) - self.q_table[state][action])
    
    def move(self, snack, other_snake):
        state = self.get_state(other_snake, snack)
        action = self.make_action(state)
        self.last_direction = (self.dirnx, self.dirny)
        if action == 0:  # Left
            self.dirnx = -1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
            
        elif action == 1:  # Right
            self.dirnx = 1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
            
        elif action == 2:  # Up
            self.dirnx = 0
            self.dirny = -1
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
            
        elif action == 3:  # Down
            self.dirnx = 0
            self.dirny = 1
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]

        for i, c in enumerate(self.body):
            p = c.pos[:]
            if p in self.turns:
                turn = self.turns[p]
                c.move(turn[0], turn[1])
                if i == len(self.body) - 1:
                    self.turns.pop(p)
            else:
                c.move(c.dirnx, c.dirny)

        new_state = self.get_state(other_snake, snack)
        return state, new_state, action

    def check_out_of_board(self):
        headPos = self.head.pos
        if headPos[0] >= ROWS - 1 or headPos[0] < 1 or headPos[1] >= ROWS - 1 or headPos[1] < 1:
            self.reset((random.randint(3, 18), random.randint(3, 18)))
            return True
        return False
        
    def get_next_position(self):
        next_x = self.head.pos[0] + self.dirnx
        next_y = self.head.pos[1] + self.dirny
        return (next_x, next_y)
    
    def calc_reward(self, snack, other_snake):
        reward = 0
        win_self, win_other = False, False

        if self.check_out_of_board():
            reward -= 300  # Punish the snake for getting out of the board
            win_other = True
            reset(self, other_snake)  # No winner


        if self.head.pos == snack.pos:
            self.addCube()
            snack = Cube(randomSnack(ROWS, self), color=(0, 255, 0))
            reward += 400  # Reward the snake for eating

        if self.head.pos in list(map(lambda z: z.pos, self.body[1:])):
            reward -= 45  # Punish the snake for hitting itself
            win_other = True
            reset(self, other_snake)  


        if self.head.pos in list(map(lambda z: z.pos, other_snake.body)):
            if self.head.pos != other_snake.head.pos:
                reward -= 25  # Punish the snake for hitting the other snake
                win_other = True
            else:
                if len(self.body) >  len(other_snake.body):
                    reward += 30  # Reward the snake for hitting the head of the other snake and being longer
                    win_self = True
                if len(self.body) > 3 + len(other_snake.body):
                    reward += 420  
                    win_self = True
                
                elif len(self.body) == len(other_snake.body):
                    win_other, win_self = False, False
                    reset(self, other_snake)  # No winner
                else:
                    reward -= 40  # Punish the snake for hitting the head of the other snake and being shorter
                    win_other = True
            reset(self, other_snake)  
        if (self.dirnx, self.dirny) != (self.last_direction[0], self.last_direction[1]):
            reward -= 35

        
        current_distance = self.get_manhattan_distance(self.head.pos, snack.pos)
        next_pos = self.get_next_position()
        next_distance = self.get_manhattan_distance(next_pos, snack.pos)

        if next_distance < current_distance:
            reward += 10  # Reward for moving closer to the snack
        else:
            reward -= 5  # Small penalty for not moving towards the snack
        return snack, reward, win_self, win_other
    
    def get_manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        
    def reset(self, pos):
        self.head = Cube(pos, color=self.color)
        self.body = []
        self.body.append(self.head)
        self.turns = {}  
        self.dirnx = 0
        self.dirny = 1

        

    def addCube(self):
        tail = self.body[-1]
        dx, dy = tail.dirnx, tail.dirny

        if dx == 1 and dy == 0:
            self.body.append(Cube((tail.pos[0] - 1, tail.pos[1]), color=self.color))
        elif dx == -1 and dy == 0:
            self.body.append(Cube((tail.pos[0] + 1, tail.pos[1]), color=self.color))
        elif dx == 0 and dy == 1:
            self.body.append(Cube((tail.pos[0], tail.pos[1] - 1), color=self.color))
        elif dx == 0 and dy == -1:
            self.body.append(Cube((tail.pos[0], tail.pos[1] + 1), color=self.color))

        self.body[-1].dirnx = dx
        self.body[-1].dirny = dy

    def draw(self, surface):
        for i, c in enumerate(self.body):
            if i == 0:
                c.draw(surface, True)
            else:
                c.draw(surface)

    def save_q_table(self, file_name):
        np.save(file_name, self.q_table)

    def get_relative_loc(self, point):
        loc = 0
        if point.pos[0] <= self.head.pos[0] and point.pos[1] <= self.head.pos[1]:
            loc = 0 # west north
        elif point.pos[0] >= self.head.pos[0] and point.pos[1] <= self.head.pos[1]:
            loc = 1 # east north
        elif point.pos[0] >= self.head.pos[0] and point.pos[1] >= self.head.pos[1]:
            loc = 2 # east south
        elif point.pos[0] <= self.head.pos[0] and point.pos[1] >= self.head.pos[1]:
            loc = 3 # west south

        return loc
       
    def check_obstacle(self,other_snake,reduced_grid):
        obstacles = [int(0)] * 10
        for key, value in reduced_grid.items():
            if value in list(map(lambda z: z.pos, self.body[1:])):
                obstacles[key] = 1
            elif value in list(map(lambda z: z.pos, other_snake.body)):
                obstacles[key] = 1
            elif value[0] == ROWS or value[0] == 0 or value[1] == ROWS or value[1] == 0:
                obstacles[key] = 1
       
        return obstacles

    def get_state(self, other_snake, snack): 
        apple = 10
        reduced_grid = {
            0: (self.head.pos[0], self.head.pos[1] - 2),
            1: (self.head.pos[0], self.head.pos[1] - 1),
            2: (self.head.pos[0] + 1, self.head.pos[1] - 1),
            3: (self.head.pos[0] - 1, self.head.pos[1] - 1),
            4: (self.head.pos[0] + 2, self.head.pos[1]),
            5: (self.head.pos[0] + 1, self.head.pos[1]),
            6: (self.head.pos[0] - 1, self.head.pos[1]),
            7: (self.head.pos[0] - 2, self.head.pos[1]),
            8: (self.head.pos[0] + 1, self.head.pos[1] + 1),
            9: (self.head.pos[0] - 1, self.head.pos[1] + 1)
        }
        
        for key, value in reduced_grid.items():
            if value == snack.pos:
                apple = key
        
        return (apple, self.get_relative_loc(snack),*(self.check_obstacle(other_snake,reduced_grid)))

    
