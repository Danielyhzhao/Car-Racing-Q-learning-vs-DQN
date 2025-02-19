# Car-Racing-Q-learning-vs-DQN

# Multi-Agent Racing Simulation with Q-Learning and DQN

This project simulates a two-car racing game using both Q-Learning and Deep Q-Network (DQN) agents. It implements a dynamic racing environment with a changing road profile, along with reinforcement learning algorithms to drive two competing cars. The simulation is rendered in real time using Pygame, and training progress is visualized with Matplotlib.

## Overview

- **Multi-Agent Environment:** Two racing cars compete on the same track.
- **Dynamic Road Generation:** The road center and the width vary as functions of the vertical position (`world_y`) to simulate realistic track undulation.
- **Q-Learning Agent:** Uses a Q-table with state discretization and an epsilon-greedy strategy.
- **DQN Agent:** Leverages a feed-forward neural network to approximate Q-values, with experience replay and target network updates.
- **Visualization:** Generates learning curves, success rate plots, convergence curves, and performance score graphs.

## Architecture & Code Structure

- **Environment (`MultiAgentRacingEnv`):**  
  Handles rendering, updating car positions, checking finishes or collisions, computing rewards, and drawing the race track.
  
- **Racing Car (`RacingCar`):**  
  Represents the physical car with properties such as position, speed, boundaries (finished or crashed), and methods for updating its state and rendering itself.
  
- **Learning Agents:**
  - **Q-Learning Agent (`QLearner`):**  
    Implements the Q-Learning algorithm with state discretization and an epsilon-greedy action selection strategy.
    
  - **DQN Agent (`DQNAgent`):**  
    Uses a neural network (built with PyTorch) to estimate Q-values. The agent also implements experience replay and periodically updates a target network to stabilize learning.
  
- **Main Function:**  
  Orchestrates training and validation episodes, updates agents, displays training diagnostics, and finally runs a demonstration of the trained agents.

## Mathematical Formulas

### Road Center Calculation

The road center is computed based on the vehicle’s \( y \)-coordinate (`world_y`) using a segmented approach. For example:

- For world_y > 5000 :

      base = 8

- For 4000 < word_y <= 5000:  

      base = 8 + (5000 - world_y)/1000 X 4


- For 3000 < world_y <= 4000:

      base = 12 - (4000 - world_y)/1000 X 4


A periodic fluctuation is then added:
  
    irregular = 0
 
    irregular = 0.8sin(world_y/100) + 0.5cos(world_y/50), if world_y > 5500 otherwise


Thus, the overall road center value becomes:
  
    road_center_value = base + irregular

This value is mapped to pixel space where the offset is computed as:

    offset = (road_center_value X CELL_SIZE + CELL_SIZE/2) - TRACK_CENTER

### Q-Learning Update

The Q-Learning agent updates its Q-table according to the rule:

    Q(s,a)←Q(s,a)+α[r+γmaxQ(s',a') - Q(s,a)]

where:
- α is the learning rate.
- γ is the discount factor.
- r is the immediate reward.
- s and s' are the current and next states.
- a and a' represent the selected and best next action respectively.

### DQN Loss Function

The DQN agent is trained by minimizing the Mean Squared Error (MSE) loss between the predicted Q-values and the target Q-values computed as:

    loss=MSE(Q(s,a),r+γmaxQtarget(s',a'))

Here, \( Q_{\text{target}} \) denotes the target network, which is periodically updated to improve training stability.

## Installation

Ensure you have Python 3 installed along with the following packages:

```bash
pip install pygame numpy torch matplotlib
```

## Running the Simulation

Run the simulation with the command:

```bash
python Car_Racing.py
```

Replace `Car_Racing.py` with the actual filename.

## Visualization

The code generates several plots during training to visualize:
- **Learning Curve:** Cumulative rewards per episode.
- **ROC Curve:** Success (finish) rate vs. episodes.
- **Convergence Curve:** Average update magnitude (for Q-Learning) or DQN loss.
- **Performance Score:** Normalized performance based on the distance traveled per step.

Each plot is generated with a large figure size (using `figsize=(16, 10)`) to clearly display all the details.

## Demo Phase

After training and validation, a demo phase runs where the trained agents race. The simulation window displays the final race, and the demo remains active for a few seconds after completion.

## Acknowledgments

This project integrates ideas from reinforcement learning, deep learning, and simulation. Special thanks to libraries such as Pygame for visualization and PyTorch for deep learning capabilities.
