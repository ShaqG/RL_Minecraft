# RL in Minecraft

A tabular Q-learning agent that learns to navigate a Minecraft maze, built on
[Project Malmo](https://github.com/microsoft/malmo) (Microsoft's AI platform for
Minecraft).

## How it works

- **State**: the agent's discrete grid position.
- **Actions**: `moveeast 1`, `movewest 1`, `movenorth 1`, `movesouth 1`.
- **Learning**: tabular Q-learning with `alpha = 0.3`, `gamma = 1.0`,
  `epsilon = 0.01`. The action choice is biased towards the goal direction, and
  illegal moves (walls, boundary, falling off the platform) are penalised so the
  table converges without exploring the whole map.
- **Reward**: reaching the fixed goal cell ends the episode.

`final.py` also has an optional Tkinter canvas that draws the Q-values while the
agent trains, which is handy for seeing the policy take shape.

## Files

| File | Purpose |
| --- | --- |
| `final.py` | The agent: mission setup, Q-table update, optional visualisation |
| `final.xml` | Malmo mission used for training |
| `final_test.xml` | Mission on a different map, used to check generalisation |

## Run

1. Install Project Malmo and launch Minecraft with the Malmo mod.
2. `python final.py`

Hyperparameters, the goal cell and the map size are constants at the top of
`TabQAgent.__init__`; change them together with the mission XML when switching
maps.
