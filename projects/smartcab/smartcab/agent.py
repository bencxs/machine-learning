import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from astropy.constants.si import alpha

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, epsilon=0, alpha=1.0, gamma=0.7):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # si3mple route planner to get next_waypoint
        # TODO: Initialize any additional variables here        
        self.trials = 0
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        
        # Initializing a Q-table dictionary, q, in the format: q[state][action] = reward
        self.q = {}
        light = ['red', 'green']
        oncoming = [None, 'forward', 'left', 'right']
        left = [0, 1] # 1 == 'forward', 0 otherwise
        waypoint_list = ['forward', 'left', 'right']
        
        dict_1 = {}   
        for l in light:
            for o in oncoming:
                for e in left:
                    for w in waypoint_list:
                        dict_1[l, o, e, w] = 0
        
        S = dict_1
        A = [None, 'forward', 'left', 'right']
        for s in S:
            self.q[s] = {}
            for a in A:
                self.q[s][a] = 2 # All Q-values are initialized to 2
        

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

        self.iteration = 0 # Resets iteration counter after each trial
        self.alpha =+ self.alpha*0.9 # Alpha decay is activated after every new trial
        self.epsilon =+ self.epsilon*0.95 # Epsilon decay is made quicker than alpha's

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        s_light = inputs['light']
        s_oncoming = inputs['oncoming']
        s_left = int(inputs['left']=='forward')
        self.state = (s_light, s_oncoming, s_left, self.next_waypoint)
        
        # TODO: Select action according to your policy
        #action = random.choice(action_list)
        action_list = [None, 'forward', 'left', 'right']
        
        if random.random() < self.epsilon: # Epsilon-greedy strategy implementation
            action = random.choice(action_list)
            
        else:
            Qvalues = [i for i in self.q[self.state].values()]
            maxQ = max(Qvalues)
            count = Qvalues.count(maxQ)
            if count > 1: # If multiple maxQ values exist, pick a random maxQ
                shortlist = [k for k in range(len(action_list)) if Qvalues[k] == maxQ]
                k = random.choice(shortlist)
            else:
                k = Qvalues.index(maxQ)
    
            action = self.q[self.state].keys()[k]
        
        Qold = self.q[self.state][action] # Initial Q-value at state s
        old_state = self.state # Initial state s

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        
        # Q(s, a) <-- Q(s, a) + alpha*[r + gamma*(max(Q(s', a')) - Q(s, a)]
        maxQnew = max([i for i in self.q[self.state].values()]) # Q-value of transition state s'
        Qupdated = Qold + self.alpha*(reward + self.gamma*(maxQnew - Qold)) # Q-learning implementation
        self.q[old_state][action] = Qupdated # Updates Q-table with new Q-values
            
        self.iteration = self.iteration + 1 # Iteration counter
        
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        
        # Output files for results collection
        # Contains data on all 100 trials
        '''with open("rewards_data.txt", "a") as myfile:
            myfile.write("Iter" + str(self.iteration) + '\t' + str(self.alpha) + '\t' + str(self.epsilon) + '\t' + str(deadline) + '\t' +  str(self.state) + '\t' + str(action) + '\t' + str(reward) + '\n')'''

        # Contains data on Q-table
        '''with open("q_table_data.txt", "a") as myfile:
            myfile.write(str(self.q) + '\n')'''

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
