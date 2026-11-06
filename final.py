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

        self.gamma = 1
        self.alpha = 0.3
        self.count = 0
        self.goal = "-9:1"
        self.patterns = patterns
        self.new_map = new_map
        self.reset_episode()

    def reset_episode(self):
        self.prev_s = self.prev_a = None
        self.last_s = None
        self.boundary = {}
        self.onboard = 0
        self.ban = -1
        self.goal_found = False
        self.steps = 0

    def calculateDir(self, curr_x, curr_y, goal_x, goal_y):
        a = None
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
        total_weight = sum(weights)
        return a, [weight / total_weight for weight in weights]

    def act(self, world_state, agent_host, current_r):
        """Credit the previous move, then send and record the next move."""
        previous_state, previous_action = self.prev_s, self.prev_a
        self.prev_s = self.prev_a = None

        block = None
        for reward, name in ((20, "block1"), (50, "block2"), (70, "block3")):
            if reward - 5 < current_r < reward + 5:
                block = name
                self.boundary[block] = 1
                current_r -= reward
                break

        action = None
        if block is None:
            if self.onboard and self.ban != -1 and previous_action is not None:
                action = previous_action ^ 1  # Paired directions are opposites.
                self.ban = previous_action
            self.boundary = {}
        elif self.new_map == 1 and previous_action is not None:
            if block not in self.patterns:
                self.q_table[previous_state][previous_action] -= 10
                action = previous_action ^ 1
            elif not self.onboard:
                self.onboard = 1
                self.q_table[previous_state][previous_action] += 10
                action = previous_action
                if self.ban == -1:
                    self.ban = previous_action

        obs = json.loads(world_state.observations[-1].text)
        x, z = float(obs[u'XPos']), float(obs[u'ZPos'])
        current_state = "%d:%d" % (int(x), int(z))
        values = self.q_table.setdefault(current_state, [0] * len(self.actions))

        if previous_action is not None:
            old_q = self.q_table[previous_state][previous_action]
            self.q_table[previous_state][previous_action] = old_q + self.alpha * (
                current_r + self.gamma * max(values) - old_q
            )

        self.drawQ(curr_x=int(x) + 20, curr_y=int(z) + 10)

        commands = None
        if action is None:
            if previous_action is not None and self.last_s == (x, z):
                action = previous_action
                jump_commands = (
                    ("turn 90", "turn 90", "jumpmove 1", "turn -90", "turn -90"),
                    ("jumpmove 1",),
                    ("turn 90", "jumpmove 1", "turn -90"),
                    ("turn -90", "jumpmove 1", "turn 90"),
                )
                commands = jump_commands[action]
            elif random.uniform(0, 1) < self.epsilon:
                action = random.randint(0, len(self.actions) - 1)
            else:
                best_value = max(values)
                action = random.choice([i for i, value in enumerate(values) if value == best_value])
            if commands is None and action == self.ban:
                action = random.choice([i for i in range(len(self.actions)) if i != self.ban])

        if self.goal:
            goal_x, goal_z = self.goal.split(":")
            bonus, weights = self.calculateDir(x, z, int(goal_x), int(goal_z))
            if bonus is not None:
                for direction in (bonus, (bonus - 1) % len(self.actions), (bonus + 1) % len(self.actions)):
                    values[direction] += 0.5 * weights[direction]

        movement = "jumpmove 1" if commands is not None else self.actions[action]
        if commands is None:
            commands = (movement,)
            time.sleep(0.2)
        try:
            for command in commands:
                agent_host.sendCommand(command)
                if command == movement:
                    self.prev_s = current_state
                    self.prev_a = action
                    self.last_s = (x, z)
                    self.steps += 1
                time.sleep(0.2 if len(commands) > 1 else 0.1)
        except RuntimeError as error:
            print("Failed to send command: %s" % error)

    def read_rewards(self, world_state):
        for error in world_state.errors:
            print("Error: %s" % error.text)
        values = [reward.getValue() for reward in world_state.rewards]
        if 100 in values:
            self.goal_found = True
        return sum(values)

    def run(self, agent_host):
        """run the agent on the world"""

        total_reward = 0
        obs = None
        self.reset_episode()
        
        is_first_action = True
        
        # main loop:
        world_state = agent_host.getWorldState()
        current_r = self.read_rewards(world_state)
        while world_state.is_mission_running:
            
            if is_first_action:
                # wait until have received a valid observation
                while True:
                    time.sleep(0.1)
                    world_state = agent_host.getWorldState()
                    current_r += self.read_rewards(world_state)
                        
                    if world_state.is_mission_running and len(world_state.observations)>0 and not world_state.observations[-1].text=="{}":
                        obs_text = world_state.observations[-1].text
                        obs = json.loads(obs_text)
                        current_s = "%d:%d" % (int(obs[u'XPos']), int(obs[u'ZPos']))
                        self.act(world_state, agent_host, current_r)
                        total_reward += current_r
                        break
                    if not world_state.is_mission_running:
                        break
                is_first_action = False
            else:
                # wait for non-zero reward
                while world_state.is_mission_running and current_r == 0:
                    time.sleep(0.1)
                    world_state = agent_host.getWorldState()
                    current_r += self.read_rewards(world_state)
                        
                # allow time to stabilise after action
                while world_state.is_mission_running:
                    time.sleep(0.1)
                    world_state = agent_host.getWorldState()
                    current_r += self.read_rewards(world_state)
                        
                    if world_state.is_mission_running and len(world_state.observations)>0 and not world_state.observations[-1].text=="{}":
                        #print("TRYING TO ACT")
                        obs_text = world_state.observations[-1].text
                        obs = json.loads(obs_text)
                        current_s = "%d:%d" % (int(obs[u'XPos']), int(obs[u'ZPos']))
                        self.act(world_state, agent_host, current_r)
                        total_reward += current_r
                        print("--- Total reward: ", total_reward)
                        
                        break
                    if not world_state.is_mission_running:
                        break
            if world_state.is_mission_running:
                current_r = 0

        # process final reward
        total_reward += current_r
        if self.goal_found:
            print("###### GOAL FOUND ######")
            for k,v in self.boundary.items():
                self.patterns[k] += 1
            #print("PATTERN: ",self.patterns)
            if obs is not None:
                self.goal = current_s
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

def print_summary(phase, rewards, successes):
    print("%s summary: %d/%d episodes found a diamond, mean reward %.1f"
          % (phase, successes, len(rewards), sum(rewards) / float(len(rewards))))


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
successes = 0
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

    print("Episode: %d moves, reward %.1f, diamond %s"
          % (agent.steps, cumulative_reward, "found" if agent.goal_found else "not found"))
    cumulative_rewards.append(cumulative_reward)
    successes += int(agent.goal_found)
    
    end_time = time.time()
    time_taken = end_time - start_time
    print("Time Taken: ",time_taken)

    # -- clean up -- #
    print("PATTERN: ",agent.patterns)
    time.sleep(0.5) # (let the Mod reset)
print_summary("Training", cumulative_rewards, successes)
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
    successes = 0
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
        
        print("Episode: %d moves, reward %.1f, diamond %s"
              % (agent.steps, cumulative_reward, "found" if agent.goal_found else "not found"))
        cumulative_rewards.append(cumulative_reward)
        successes += int(agent.goal_found)
        if agent.goal_found:
            time.sleep(30)
            break

        # -- clean up -- #
        print("PATTERN: ",agent.patterns)
        time.sleep(0.5) # (let the Mod reset)
    else:
        print("No diamond found on the test map.")
    print_summary("Test", cumulative_rewards, successes)
else:
    print("No block patterns recorded in training. Skipping the test map.")
