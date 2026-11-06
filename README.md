# Q-learning in Minecraft

A tabular Q-learning agent that searches for diamond blocks in two Minecraft maps built with [Project Malmo](https://github.com/microsoft/malmo).

## Run a mission

1. Follow the Project Malmo setup instructions and start its Minecraft client. Use a Python environment that can import `MalmoPython`, `future`, and Tkinter.
2. From the repository root, run `python final.py`.

Watch the agent move in Minecraft and the estimated action values change in the **Q-table** window. The terminal prints cumulative rewards for completed episodes. Touching a diamond block earns +100 and ends the episode. The terminal prints `GOAL FOUND` when the agent detects a successful finish.

## How it works

- The agent uses its integer X/Z position as the state and chooses among four cardinal movement actions.
- Q-learning uses a learning rate of 0.3, a discount factor of 1.0, and an exploration rate of 0.01. The agent also biases action values toward the goal and uses jump commands when movement stalls.
- Training runs on [`final.xml`](final.xml). If it records block patterns during successful training runs, [`final.py`](final.py) starts a fresh agent on [`final_test.xml`](final_test.xml) and passes those patterns to it. The second agent starts with an empty Q-table and tries up to ten episodes, stopping early when it finds a diamond. If training records no block patterns, the script skips the test map.
