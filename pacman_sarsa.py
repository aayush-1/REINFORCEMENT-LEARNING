import time
import random
import collections
import matplotlib.pyplot as plt
import numpy as np
import os
x=0

class Environment(object):
    def __init__(self, size, density):
        self.size = size
        self.density = density
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0),(0,0)] # left, down, right, up
        
    def initialize(self):
        
        locations = list()
        for r in range(1,self.size-1):
            for c in range(1,self.size-1):
                locations.append((r, c))
        
        random.shuffle(locations)
        self.pacman = locations.pop()
        
        self.pellets = set()
        for count in range(self.density):
            self.pellets.add(locations.pop())
            
        self.new_ghost()
        self.next_reward = 0
        self.pellets_consumed=0
    def new_ghost(self):
        (r, c) = self.pacman
        locations = [(r, 0), (0, c), (r, self.size-1), (self.size-1, c)]
        choice = random.choice(range(len(locations)))
        self.ghost = locations[choice]
        self.ghost_action = self.directions[choice]
    
    def display(self):
        for r in range(self.size):
            for c in range(self.size):
                if (r,c) == self.ghost:
                    print( 'G' ,end =" "),
                elif (r,c) == self.pacman:
                    print( 'O',end =" "),
                elif (r,c) in self.pellets:
                    print( '.',end =" "),
                elif r == 0 or r == self.size-1:
                    print( 'X',end =" "),
                elif c == 0 or c == self.size-1:
                    print( 'X',end =" "),
                else:
                    print( ' ',end =" "),
            print()
        print()

        print("pellets_consumed SARSA: ", self.pellets_consumed)
    def actions(self):
        if self.terminal():
            return None
        else:
            return self.directions

    def terminal(self):
        if self.next_reward == -100:
            return True
        elif len(self.pellets) == 0:
            locations = list()
            for r in range(1,self.size-1):
                for c in range(1,self.size-1):
                    locations.append((r, c))
            
            random.shuffle(locations)
            locations.remove(self.pacman)            
            for count in range(self.density):
                self.pellets.add(locations.pop())
            return False
        else:
            return False
    
    def reward(self):
        return self.next_reward
        
    def update(self, action):
        
        pacman = self.pacman
        ghost = self.ghost
        
        # Pacman moves as chosen
        (r, c) = self.pacman
        (dr, dc) = action
        self.pacman = (r+dr, c+dc)

        # Ghost moves in its direction
        (r, c) = self.ghost
        (dr, dc) = self.ghost_action
        self.ghost = (r+dr, c+dc)
        
        # Ghost is replaced when it leaves
        (r, c) = self.ghost
        if r == 0 or r == self.size-1:
            self.new_ghost()
        elif c == 0 or c == self.size-1:
            self.new_ghost()
        
        (r,c) = self.pacman
        (gr,gc) = self.ghost
        
        # Negative reward for hitting the ghost
        if self.pacman == self.ghost:
            self.next_reward = -100
        elif (pacman, ghost) == (self.ghost, self.pacman):
            self.next_reward = -100
        
        # Negative reward for hitting a wall
        elif r == 0 or r == self.size-1:
            self.next_reward = -100
        elif c == 0 or c == self.size-1:
            self.next_reward = -100
        
        # Positive reward for consuming a pellet
        elif self.pacman in self.pellets:
            self.next_reward = 10
            self.pellets_consumed=self.pellets_consumed+1
            self.pellets.remove(self.pacman)
        else:
            self.next_reward = -1

        

    def state(self):
        s = dict()

        s['pellet position'] = sorted(self.pellets)
        s['ghost position'] = self.ghost
        s['pacman position'] = self.pacman

        
        
        return s

class Agent(object):

    def __init__(self):
        self.w = collections.defaultdict(float) 
        self.epsilon = 0.05 # Exploration rate
        self.gamma = 0.99 # Discount factor
        self.q_table=dict()
        self.alpha = 0.1 # Learning rate
        self.ACTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0),(0,0)]
    def choose(self, s, actions):
        p = random.random()
        if p < self.epsilon:
            return random.choice(actions)
        else:
            return self.policy(s, actions)

    def policy(self, s, actions):
        max_value = max([self.Q(s,a) for a in range(5)])
        max_actions = [actions[a] for a in range(5) if self.Q(s,a) == max_value]
        return random.choice(max_actions)

    def Q(self, s, a=None):
        s=str(s)
        
        if s not in self.q_table:
            self.q_table[s] = np.zeros(len(self.ACTIONS))

        if a is None:
            return self.q_table[s]

        return self.q_table[s][a]
    
    def observe(self, s, a, sp, r, actions):
        ap=self.choose(sp,[(0, 1), (1, 0), (0, -1), (-1, 0),(0,0)])
        ap=self.ACTIONS.index(ap)
        a=self.ACTIONS.index(a)
        self.q_table[str(s)][a]=self.Q(s,a)+ self.alpha*(r+self.gamma*self.Q(sp,ap) - self.Q(s,a))


def main():

    environment = Environment(6,1)
    agent = Agent()
    pellets_consumed=[]
    for episode in range(10000):
        environment.initialize()
        while not environment.terminal():
            
            s = environment.state()
            actions = environment.actions()
            a = agent.choose(s, actions)
            environment.update(a)
            
            sp = environment.state()
            r = environment.reward()
            actions = environment.actions()
            agent.observe(s, a, sp, r, actions)

        print(len(agent.q_table))
    environment.initialize()
    environment.display()
    while not environment.terminal():
        
        s = environment.state()
        actions = environment.actions()
        a = agent.policy(s, actions)
        
        environment.update(a)
        time.sleep(0.25)
        os.system('clear')
        environment.display()

if __name__ == '__main__':
    main()
