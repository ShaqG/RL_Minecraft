from future import standard_library
standard_library.install_aliases()
from builtins import range
from builtins import object
import MalmoPython
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict, deque
if sys.version_info[0] == 2:
    # Workaround for https://github.com/PythonCharmers/python-future/issues/262
    import Tkinter as tk
else:
    import tkinter as tk

class TabQAgent(object):
    """Tabular Q-learning agent for discrete state/action spaces."""

    def __init__(self, patterns, new_map):
        self.epsilon = 0.01 
        # back, forward: west, right, left
        self.actions = ["moveeast 1", "movewest 1", "movenorth 1", "movesouth 1" ]
        self.q_table = {}
        self.canvas = None
        self.root = None

        self.last_s = (0.0,0.0) 
        self.last_a = 0
        self.gamma = 1
        self.alpha = 0.3
        self.count = 0
        self.goal = "-9:1"
        self.alti = 33
        self.patterns = patterns
        self.boundary = {}
        self.onboard = 0
        self.ban = -1
        self.test_mode = False
        self.new_map = new_map

    def calculateDir(self, curr_x, curr_y, goal_x, goal_y):
        #print("------- CALCULATING -------")
        
        #prob = [0.25,0.50,0.75]
        weights = [0.25,0.25,0.25,0.25]
        if curr_x  - goal_x < 0:
            #prob[0] += 0.1
            a = 0
            weights[0] += 0.5
        elif curr_x - goal_x > 0:
            #prob[0] -= 0.1
            a = 1
            weights[1] += 0.5
        if curr_y  - goal_y < 0:
            #prob[2] -= 0.1
            a = 3
            weights[3] += 0.5
        elif curr_y - goal_y > 0:
            #prob[2] += 0.1
            a = 2
            weights[2] += 0.5
        #print(prob)
        weights = [w/sum(weights) for w in weights]
        return a, weights
        p = random.random()
        if p <= prob[0]:
            return 0, weights
        elif p > prob[0] and p <= prob[1]:
            return 1, weights
        elif p > prob[1] and p <= prob[2]:
            return 2, weights
        else:
            return 3, weights


    def act(self, world_state, agent_host, current_r ):
        """take 1 action in response to the current world state"""
        action_sent = 0
        if current_r > 15 and current_r < 25:
            block = "block1"
            if block not in self.boundary:
                self.boundary[block] = 1
            # else:
            #     self.boundary[block] = 1-self.boundary[block]       
                #print(self.boundary[block])         
                
            current_r -= 20
            #print("------Wool Detected")
        elif current_r > 45 and current_r < 55:
            block = "block2"
            if block not in self.boundary:
                self.boundary[block] = 1
            # else:
            #     self.boundary[block] = 1-self.boundary[block]
                
            current_r -= 50
            #print("------Sand Detected")

        elif current_r > 65 and current_r < 75:
            block = "block3"
            if block not in self.boundary:
                self.boundary[block] = 1
            # else:
            #     self.boundary[block] = 1-self.boundary[block]
                
            current_r -= 70
            #print("------Iron Detected")
        else:
            block = ""
            if self.onboard == 1 and self.ban != -1:
                print("Leaving... Send back")
                #self.q_table[self.prev_s][self.last_a] -= 10
                if self.last_a == 0 or self.last_a == 2:
                    agent_host.sendCommand(self.actions[self.last_a+1])
                else:
                    agent_host.sendCommand(self.actions[self.last_a-1])
                action_sent = 1
                self.ban = self.last_a
            self.boundary = {}
        # print("Banned Action: ",self.ban)
        #print("-----boundary: ",self.boundary)
        
        if block != "" and self.new_map == 1:
            if block not in self.patterns:
                print("reversing...")
                self.q_table[self.prev_s][self.last_a] -= 10
                if self.last_a == 0 or self.last_a == 2:
                    agent_host.sendCommand(self.actions[self.last_a+1])
                else:
                    agent_host.sendCommand(self.actions[self.last_a-1])
                action_sent = 1
                
            elif self.onboard == 0: 
                print("Approaching...")
                self.onboard = 1
                self.q_table[self.prev_s][self.last_a] += 10
                agent_host.sendCommand(self.actions[self.last_a])
                if self.ban == -1:
                    self.ban = self.last_a
                action_sent = 1

        
        obs_text = world_state.observations[-1].text
        obs = json.loads(obs_text) # most recent observation
      
        # location get
        #current_s = "%d:%d:%d" % (int(obs[u'XPos']), int(obs[u'ZPos']), int(obs[u'YPos']))
        current_s = "%d:%d" % (int(obs[u'XPos']), int(obs[u'ZPos']))
        if current_s not in self.q_table:
            #print("--- q_table updated")
            self.q_table[current_s] = ([0] * len(self.actions))

        # update Q values
        if self.prev_s is not None and self.prev_a is not None:
            #print("--- q_value updated")
            old_q = self.q_table[self.prev_s][self.prev_a]
            self.q_table[self.prev_s][self.prev_a] = old_q + self.alpha * (current_r
                + self.gamma * max(self.q_table[current_s]) - old_q)
            #self.updateQTable( current_r, current_s )

        self.drawQ( curr_x = int(obs[u'XPos'])+20, curr_y = int(obs[u'ZPos'])+10)

        if random.uniform(0, 1) < self.epsilon:
            a = random.randint(0,len(self.actions) - 1)
        else:
            v = max(self.q_table[current_s])
            rand_list = []
            for i in range(0,len(self.actions)):
                if self.q_table[current_s][i] == v:
                    rand_list.append(self.actions[i])
            a = self.actions.index(rand_list[random.randint(0,len(rand_list) - 1)])

        # try to send the selected action, only update prev_s if this succeeds
        try:
            #print("--- sending command")
            new_x = float(obs[u'XPos'])
            new_y = float(obs[u'ZPos'])
            #print(new_x,new_y)
            if self.test_mode:
                curr_h = float(obs[u'YPos'])
                if curr_h < self.alti:
                    # Climb mode
                    pass
                else:
                    # Dig mode
                    pass

            if self.last_s == (new_x,new_y):
                if self.last_a == 0:
                    agent_host.sendCommand("turn 90")
                    time.sleep(0.2)
                    agent_host.sendCommand("turn 90")
                    time.sleep(0.2)
                    agent_host.sendCommand("jumpmove 1")
                    time.sleep(0.2)
                    agent_host.sendCommand("turn -90")
                    time.sleep(0.2)
                    agent_host.sendCommand("turn -90")
                    #current_r += 0.5
                elif self.last_a == 1:
                    agent_host.sendCommand("jumpmove 1")
                    #current_r += 0.5
                elif self.last_a == 2:
                    agent_host.sendCommand("turn 90")
                    time.sleep(0.2)
                    agent_host.sendCommand("jumpmove 1")
                    #current_r += 0.5
                    time.sleep(0.2)
                    agent_host.sendCommand("turn -90")
                elif self.last_a == 3:
                    agent_host.sendCommand("turn -90")
                    time.sleep(0.2)
                    agent_host.sendCommand("jumpmove 1") 
                    #current_r += 0.5
                    time.sleep(0.2)
                    agent_host.sendCommand("turn 90")
                action_sent = 1
            
            if self.goal != "":
                goal_x = self.goal.split(":")[0]
                goal_y = self.goal.split(":")[1]
                bonus, weights = self.calculateDir(new_x,new_y,int(goal_x),int(goal_y))
                left = bonus-1 if bonus > 0 else 3
                right = bonus+1 if bonus < 3 else 0

                self.q_table[current_s][bonus] += (0.5 * weights[bonus])
                self.q_table[current_s][left] += (0.5 * weights[left])
                self.q_table[current_s][right] += (0.5 * weights[right])

            # if self.cross == 1:
            #     agent_host.sendCommand(self.actions[self.last_a])
            #     time.sleep(0.2)
            #     curr_r = 0
            #     for reward in world_state.rewards:
            #             curr_r += reward.getValue()
            #     while curr_r > 0:
            #         agent_host.sendCommand(self.actions[self.last_a])
            #         time.sleep(0.2)
            #         curr_r = 0
            #         for reward in world_state.rewards:
            #             curr_r += reward.getValue()
                
            #    self.noJump = 1
            #    self.cross = 0
            # else:
            #     self.noJump = 0
            time.sleep(0.2)
            if action_sent == 0:
                print("Banned Action: ",self.ban)
                if a != self.ban:
                    agent_host.sendCommand(self.actions[a])
                else:
                    while a == self.ban:
                        a = random.randint(0,len(self.actions) - 1)
                    agent_host.sendCommand(self.actions[a])
                self.last_a = a
            time.sleep(0.1)
            
            # if self.count % 2 == 0 :
            #agent_host.sendCommand("movewest 1")
            #time.sleep(0.1)
            
            self.last_s = (new_x,new_y)
            
            
            #self.actions[a]
            self.prev_s = current_s
            self.prev_a = a

        except RuntimeError as e:
            print("Failed to send command: %s" % e)
        
        return current_r

    def run(self, agent_host):
        """run the agent on the world"""

        total_reward = 0
        
        self.prev_s = None
        self.prev_a = None
        
        is_first_action = True
        
        # main loop:
        world_state = agent_host.getWorldState()
        #print("state: ",world_state)
        while world_state.is_mission_running:

            current_r = 0
            
            if is_first_action:
                # wait until have received a valid observation
                while True:
                    time.sleep(0.1)
                    world_state = agent_host.getWorldState()
                    for error in world_state.errors:
                        print("Error: %s" % error.text)
                    for reward in world_state.rewards:
                        current_r += reward.getValue()
                        
                    if world_state.is_mission_running and len(world_state.observations)>0 and not world_state.observations[-1].text=="{}":
                        obs_text = world_state.observations[-1].text
                        obs = json.loads(obs_text)
                        current_s = "%d:%d" % (int(obs[u'XPos']), int(obs[u'ZPos']))
                        total_reward += self.act(world_state, agent_host, current_r)
                        break
                    if not world_state.is_mission_running:
                        break
                is_first_action = False
            else:
                # wait for non-zero reward
                while world_state.is_mission_running and current_r == 0:
                    time.sleep(0.1)
                    world_state = agent_host.getWorldState()
                    for error in world_state.errors:
                        print("Error: %s" % error.text)
                    for reward in world_state.rewards:
                        
                        current_r += reward.getValue()
                        
                # allow time to stabilise after action
                while True:
                    time.sleep(0.1)
                    world_state = agent_host.getWorldState()
                    for error in world_state.errors:
                        print("Error: %s" % error.text)
                    for reward in world_state.rewards:
                        current_r += reward.getValue()
                        
                    if world_state.is_mission_running and len(world_state.observations)>0 and not world_state.observations[-1].text=="{}":
                        #print("TRYING TO ACT")
                        obs_text = world_state.observations[-1].text
                        obs = json.loads(obs_text)
                        current_s = "%d:%d" % (int(obs[u'XPos']), int(obs[u'ZPos']))
                        total_reward += self.act(world_state, agent_host, current_r)
                        print("--- Total reward: ", total_reward)
                        
                        break
                    if not world_state.is_mission_running:
                        break
        #print("Q-Table: ", self.q_table)
        # process final reward
        total_reward += current_r
        if current_r > 90:
            print("###### GOAL FOUND ######")
            for k,v in self.boundary.items():
                self.patterns[k] += 1
            #print("PATTERN: ",self.patterns)
            self.goal = current_s
            self.alti = int(obs[u'YPos'])
            #print("---- alti: ",self.alti)
        #print("--- Total reward: ", total_reward)

        # update Q values
        if self.prev_s is not None and self.prev_a is not None:
            #self.updateQTableFromTerminatingState( current_r )
            old_q = self.q_table[self.prev_s][self.prev_a]
            self.q_table[self.prev_s][self.prev_a] = old_q + self.alpha * ( current_r - old_q )
            
        self.drawQ()
        
        return total_reward
        
    def drawQ( self, curr_x=None, curr_y=None):
        scale = 20
        world_x = 40
        world_y = 22
        if self.canvas is None or self.root is None:
            self.root = tk.Tk()
            self.root.wm_title("Q-table")
            self.canvas = tk.Canvas(self.root, width=world_x*scale, height=world_y*scale, borderwidth=0, highlightthickness=0, bg="black")
            self.canvas.grid()
            self.root.update()
        self.canvas.delete("all")
        action_inset = 0.1
        action_radius = 0.1
        curr_radius = 0.2
        # back, forw, right, left
        action_positions = [ ( 1-action_inset, 0.5 ), ( action_inset, 0.5 ),( 0.5, action_inset ), ( 0.5, 1-action_inset ) ]
        # (NSWE to match action order)
        min_value = -20
        max_value = 20
        for x in range(world_x):
            for y in range(world_y):
                s = "%d:%d" % (x-20,y-10)
                self.canvas.create_rectangle( x*scale, y*scale, (x+1)*scale, (y+1)*scale, outline="#fff", fill="#000")
                for action in range(4):
                    if not s in self.q_table:
                        continue
                    value = self.q_table[s][action]
                    color = int( 255 * ( value - min_value ) / ( max_value - min_value )) # map value to 0-255
                    color = max( min( color, 255 ), 0 ) # ensure within [0,255]
                    color_string = '#%02x%02x%02x' % (255-color, color, 0)
                    self.canvas.create_oval( (x + action_positions[action][0] - action_radius ) *scale,
                                             (y + action_positions[action][1] - action_radius ) *scale,
                                             (x + action_positions[action][0] + action_radius ) *scale,
                                             (y + action_positions[action][1] + action_radius ) *scale, 
                                             outline=color_string, fill=color_string )
        if curr_x is not None and curr_y is not None:
            self.canvas.create_oval( (curr_x + 0.5 - curr_radius ) * scale, 
                                     (curr_y + 0.5 - curr_radius ) * scale, 
                                     (curr_x + 0.5 + curr_radius ) * scale, 
                                     (curr_y + 0.5 + curr_radius ) * scale, 
                                     outline="#fff", fill="#fff" )
        self.root.update()


if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import functools
    print = functools.partial(print, flush=True)

agent = TabQAgent(defaultdict(int),0)
agent_host = MalmoPython.AgentHost()

#agent_host.sendCommand('attack 1')
try:
    agent_host.parse( sys.argv )
except RuntimeError as e:
    print('ERROR:',e)
    print(agent_host.getUsage())
    exit(1)

# -- set up the mission -- #
mission_file = './final.xml'
with open(mission_file, 'r') as f:
    print("Loading mission from %s" % mission_file)
    mission_xml = f.read()
    my_mission = MalmoPython.MissionSpec(mission_xml, True)

# # add 20% holes for interest
# for x in range(1,4):
#     for z in range(1,13):
#         if random.random()<0.1:
#             my_mission.drawBlock( x,45,z,"lava")

max_retries = 3
if agent_host.receivedArgument("test"):
    num_repeats = 1
else:
    num_repeats = 30

cumulative_rewards = []
for i in range(num_repeats):
    start_time = time.time()
    print()
    print('Repeat %d of %d' % ( i+1, num_repeats ))
    
    my_mission_record = MalmoPython.MissionRecordSpec()

    for retry in range(max_retries):
        try:
            agent_host.startMission( my_mission, my_mission_record )
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission:",e)
                exit(1)
            else:
                time.sleep(2.5)

    print("Waiting for the mission to start", end=' ')
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        print(".", end="")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:",error.text)
    print()

    # -- run the agent in the world -- #
    cumulative_reward = agent.run(agent_host)
    agent.count += 1

    print('Cumulative reward: %d' % cumulative_reward)
    cumulative_rewards += [ cumulative_reward ]
    
    end_time = time.time()
    time_taken = end_time - start_time
    print("Time Taken: ",time_taken)

    # -- clean up -- #
    print("PATTERN: ",agent.patterns)
    agent.boundary = {}
    time.sleep(0.5) # (let the Mod reset)
saved_patterns = agent.patterns
if saved_patterns != {}:
    print("Training Done.")
    print()
    print("Waiting for the client to reset...")
    time.sleep(2.0)
    print("Start Testing...")

    agent = TabQAgent(saved_patterns,1)
    agent_host = MalmoPython.AgentHost()

    #agent_host.sendCommand('attack 1')
    try:
        agent_host.parse( sys.argv )
    except RuntimeError as e:
        print('ERROR:',e)
        print(agent_host.getUsage())
        exit(1)
    mission_file = './final_test.xml'
    with open(mission_file, 'r') as f:
        print("Loading mission from %s" % mission_file)
        mission_xml = f.read()
        my_mission = MalmoPython.MissionSpec(mission_xml, True)

    # # add 20% holes for interest
    # for x in range(1,4):
    #     for z in range(1,13):
    #         if random.random()<0.1:
    #             my_mission.drawBlock( x,45,z,"lava")

    num_repeats = 10

    cumulative_rewards = []
    for i in range(num_repeats):

        print()
        print('Repeat %d of %d' % ( i+1, num_repeats ))
        
        my_mission_record = MalmoPython.MissionRecordSpec()

        for retry in range(max_retries):
            try:
                agent_host.startMission( my_mission, my_mission_record )
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:",e)
                    exit(1)
                else:
                    time.sleep(2.5)

        print("Waiting for the mission to start", end=' ')
        world_state = agent_host.getWorldState()
        while not world_state.has_mission_begun:
            print(".", end="")
            time.sleep(0.1)
            world_state = agent_host.getWorldState()
            for error in world_state.errors:
                print("Error:",error.text)
        print()

        # -- run the agent in the world -- #
        cumulative_reward = agent.run(agent_host)
        
        print('Cumulative reward: %d' % cumulative_reward)
        cumulative_rewards += [ cumulative_reward ]
        if agent.goal != "":
            time.sleep(30)
            break

        # -- clean up -- #
        print("PATTERN: ",agent.patterns)
        agent.boundary = {}
        time.sleep(0.5) # (let the Mod reset)
    else:
        print("Training Failed.")
