import time
import random
import collections
import numpy as np
import itertools
import matplotlib.pyplot as plt
import os

class Environment(object):

    def __init__(self):
        self.combinations=[[1,2,3],[4,5,6],[7,8,9],[1,4,7],[2,5,8],[3,6,9],[1,5,9],[3,5,7]]

    def initialize(self):
        
        self.locations = list()
        for r in range(1,10):
                self.locations.append(r)
        self.agent=0
        self.opponent=0
        self.next_reward = 0
        self.X=list()
        self.O=list()
    



    def display(self):
        """Print the environment."""

        x=1
        for r in range(3):
            for c in range(3):
                if x in self.X:
                    print ('X |',end =" "),
                elif x in self.O:
                    print ('O |',end =" "),
                else:
                    print ('  |',end =" "),
                x=x+1
            print()
            print("__  __  __")
        print()
    
    def actions(self):
        if self.terminal():
            return None
        else:
            return self.locations

    def terminal(self):
        if self.next_reward == -1 or self.next_reward==1 or len(self.locations)==0:
            return True
        else:
            return False
    
    def reward(self):
        return self.next_reward

    def random_agent(self):
        return np.random.choice(self.locations)

    def safe_agent(self):
        self.O=list(np.sort(self.O))
        A=list(itertools.combinations(self.O, 2))
        A=[list(i) for i in A]
        for i in A:
            for j in self.locations:
                B=[]
                B.append(j)
                C=i+B
                if list(np.sort(C)) in self.combinations:
                    return j
        self.X=list(np.sort(self.X))
        A=list(itertools.combinations(self.X, 2))
        A=[list(i) for i in A]
        for i in A:
            for j in self.locations:
                B=[]
                B.append(j)
                C=i+B
                if list(np.sort(C)) in self.combinations:
                    return j

        return np.random.choice(self.locations)

    def opponent_act(self,a):
        #random agent
        if(a==0):
            action=self.random_agent()
        #safe agent
        if(a==1):
            action=self.safe_agent()
        if(a==2):
            if(np.random.rand()<0.5):
                action=self.random_agent()
            else:
                action=self.safe_agent()
        
        self.O.append(action)
        self.O=list(np.sort(self.O))
        self.locations.remove(action)
        A=list(itertools.combinations(self.O, 3))
        A=[list(i) for i in A]
        for i in A: 
            if i in self.combinations:
                self.next_reward=-1
                # print("reward ",self.next_reward)
                # print("action", action)
                # print("X locations",self.X)
                # print("O locations", self.O)
                break
        self.opponent=action
        return action


        
    def update(self, action):
        """Adjust the environment given the agent's choice of action."""
        self.agent=action
        self.locations.remove(action)
        self.X.append(action)
        self.X=list(np.sort(self.X))

        A=list(itertools.combinations(self.X, 3))
        A=[list(i) for i in A]  
        for i in A: 
            if i in self.combinations:
                self.next_reward=1
                # print("reward ",self.next_reward)
                # print("action", action)
                # print("X locations",self.X)
                # print("O locations", self.O)
                break
        

    def state(self):
        s = dict()
        s['x'] =list(np.sort(self.X)) 
        s['O']=list(np.sort(self.O))
        
        return s

class Agent(object):

    def __init__(self):
        self.w = collections.defaultdict(float)
        self.epsilon = 0.9
        # exploration decays by this factor every episode
        self.epsilon_decay = 0.9
        # in the long run, 10% exploration, 90% exploitation
        self.epsilon_min = 0.1
        self.gamma = 0.99 # Discount factor
        self.alpha = 0.05 # Learning rate
        self.q_table=dict()

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def choose(self, s, actions):
        p = random.random()
        if p < self.epsilon:
            return random.choice(actions)
        else:
            return self.policy(s, actions)

    def policy(self, s, actions):
        max_value = max([self.Q(s,a) for a in actions])
        max_actions = [a for a in actions if self.Q(s,a) == max_value]
        return random.choice(max_actions)

    def Q(self, s, a=None):
        s=str(s)
        
        if s not in self.q_table:
            self.q_table[s] = np.zeros(9)

        if a is None:
            return self.q_table[s]

        return self.q_table[s][a-1]
    
    def observe(self, s, a, sp, r, actions):
        if actions==None:
            ap=8
        else:    
            ap=self.policy(sp,actions)
        self.q_table[str(s)][a-1]=self.Q(s,a)+ self.alpha*(r+self.gamma*self.Q(sp,ap) - self.Q(s,a))

def main():    
    environment = Environment()
    agent = Agent()
    win=0
    loss=0
    draw=0

    wins=[]
    losses=[]
    draws=[]
    for episode in range(20000):
        if(episode%200==0):
            wins_check=0
            loss_check=0
            draw_check=0

            for i in range(100):
                environment.initialize()
                turn=np.random.randint(2)
                r=0
                while not environment.terminal():
                    if(turn==0):
                        s_1=environment.state()
                        a_1=environment.opponent_act(0)
                        sp_1=environment.state()
                        if environment.terminal():
                            r = environment.reward()
                            break;
                        s = environment.state()
                        actions = environment.actions()
                        a = agent.policy(s, actions)
                        environment.update(a)
                        r = environment.reward()
                    else:
                        s = environment.state()
                        actions = environment.actions()
                        a = agent.policy(s, actions)
                        environment.update(a)
                        r = environment.reward()
                        if environment.terminal():
                            break
                        s_1=environment.state()
                        a_1=environment.opponent_act(0)
                        sp_1=environment.state()
                        r_1 = environment.reward() 
                if(r==1):
                    wins_check+=1
                elif(r==-1):
                    loss_check+=1
                elif(r==0):
                    draw_check+=1
            wins.append(wins_check)
            losses.append(loss_check)
            draws.append(draw_check)


        environment.initialize()
        # environment.display()
        turn=np.random.randint(2)
        while not environment.terminal():
            if(turn==0):
                s_1=environment.state()
                a_1=environment.opponent_act(0)
                sp_1=environment.state()
                actions_1=environment.actions()
                if environment.terminal():
                    r_1 = environment.reward()
                    A=[]
                    agent.observe(s,a,s_1,-1,A.append(a_1))
                    agent.observe(s_1, a_1, sp_1, +1, actions_1)
                    break
                else:
                    agent.observe(s_1, a_1, sp_1, 0, actions_1)
                s = environment.state()
                actions = environment.actions()
                a = agent.choose(s, actions)
                environment.update(a)
                
                sp = environment.state()
                r = environment.reward()
                actions = environment.actions()
                agent.observe(s, a, sp, r, actions)


            else:
                s = environment.state()
                actions = environment.actions()
                a = agent.choose(s, actions)
                environment.update(a)
                
                sp = environment.state()
                r = environment.reward()
                actions = environment.actions()
                agent.observe(s, a, sp, r, actions)
                if environment.terminal():
                    break
                s_1=environment.state()
                a_1=environment.opponent_act(0)
                sp_1=environment.state()
                r_1 = environment.reward()

                actions_1=environment.actions()
                if environment.terminal():
                    A=[]
                    agent.observe(s,a,s_1,-1,A.append(a_1))
                    agent.observe(s_1, a_1, sp_1, +1, actions_1)
                else:
                    agent.observe(s_1, a_1, sp_1, 0, actions_1)
        if r==1:
            win+=1

        elif r_1==-1:
            loss+=1
        elif r==0   :
            draw+=1
        agent.update_epsilon()

    print("\nStatistics while training---------------------------------")
    print("win-- ",win)
    print("loss-- ",loss)
    print("draw-- ",draw)

    plt.plot(range(len(wins)),wins,label = "wins")
    plt.plot(range(len(losses)),losses,label = "loss")
    plt.plot(range(len(draws)),draws,label = "draws")
    plt.show()





#################################### Testing code #################################################
    wins_test=0
    loss_test=0
    draw_test=0
    for i in range(1000):
        environment.initialize()
        # time.sleep(0.5)
        # os.system('clear')
        # environment.display()
        turn=np.random.randint(2)
        r=0
        while not environment.terminal():
            if(turn==0):
                s_1=environment.state()
                a_1=environment.opponent_act(0)
                # time.sleep(0.5)
                # os.system('clear')
                # environment.display()
                sp_1=environment.state()
                if environment.terminal():
                    r = environment.reward()
                    break;
                s = environment.state()
                actions = environment.actions()
                a = agent.policy(s, actions)
                
                environment.update(a)
                r = environment.reward()
                # time.sleep(0.5)
                # os.system('clear')
                # environment.display()
            else:
                s = environment.state()
                actions = environment.actions()
                a = agent.policy(s, actions)
                environment.update(a)
                # time.sleep(0.5)
                # os.system('clear')
                # environment.display()
                r = environment.reward()
                if environment.terminal():
                    break
                s_1=environment.state()
                a_1=environment.opponent_act(0)
                sp_1=environment.state()
                r = environment.reward()
                # time.sleep(0.5)
                # os.system('clear')
                # environment.display()


        if r==1:
            wins_test+=1
        elif r==-1:
            loss_test+=1
        elif r==0   :
            draw_test+=1
    print("Learned through Random Agent for 20000 epochs")
    print("\nStatistics while testing against Random Agent ---------------------------------")
    print("win-- ",wins_test)
    print("loss-- ",loss_test)
    print("draw-- ",draw_test)

    wins_test=0
    loss_test=0
    draw_test=0
    for i in range(1000):
        environment.initialize()
        # time.sleep(0.5)
        # os.system('clear')
        # environment.display()       
        turn=np.random.randint(2)
        r=0
        while not environment.terminal():
            if(turn==0):
                s_1=environment.state()
                a_1=environment.opponent_act(1)
                # time.sleep(0.5)
                # os.system('clear')
                # environment.display()
                sp_1=environment.state()
                if environment.terminal():
                    r = environment.reward()
                    break;
                s = environment.state()
                actions = environment.actions()
                a = agent.policy(s, actions)
                
                environment.update(a)
                r = environment.reward()
                # time.sleep(0.5)
                # os.system('clear')
                # environment.display()
            else:
                s = environment.state()
                actions = environment.actions()
                a = agent.policy(s, actions)
                environment.update(a)
                # time.sleep(0.5)
                # os.system('clear')
                # environment.display()
                r = environment.reward()
                if environment.terminal():  
                    break
                s_1=environment.state()
                a_1=environment.opponent_act(1)
                sp_1=environment.state()
                r = environment.reward()
                # time.sleep(0.5)
                # os.system('clear')
                # environment.display()

        if r==1:
            wins_test+=1
        elif r==-1:
            loss_test+=1
        elif r==0   :
            draw_test+=1

    print("\nStatistics while testing against Safe Agent ---------------------------------")
    print("win-- ",wins_test)
    print("loss-- ",loss_test)
    print("draw-- ",draw_test)       

if __name__ == '__main__':
    main()