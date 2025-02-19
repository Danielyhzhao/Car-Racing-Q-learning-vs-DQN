#!/usr/bin/env python3

import pygame
import sys
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import matplotlib.pyplot as plt

# 0: No specific action; 1: Acceleration; 2: Deceleration; 3: Move left; 4: Move right
ACTIONS = [0, 1, 2, 3, 4]

# Global Constants and Grid Parameters
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
CELL_SIZE = 50  # Grid cell is 50x50 pixels

LAP_DISTANCE = 6000  # 6000 pixels → 120 rows
# Calculate the number of cells the track has in the vertical direction versus the number of cells in the horizontal direction by integer division.
GRID_ROWS = int(LAP_DISTANCE // CELL_SIZE)
GRID_COLS = int(SCREEN_WIDTH // CELL_SIZE)

# Road width classes (in half-widths)
WIDE_ROAD_HALF_WIDTH = 120  # Wide: overall 240px.
MEDIUM_ROAD_HALF_WIDTH = 90  # Medium: overall 180px.
NARROW_ROAD_HALF_WIDTH = 60  # Narrow: overall 120px.
DEFAULT_ROAD_HALF_WIDTH = MEDIUM_ROAD_HALF_WIDTH

# The horizontal position of the centre of the track in the screen, here set to half the width of the screen.
TRACK_CENTER = SCREEN_WIDTH // 2


# Road Center and Offset Function Definitions
# Add a periodic variation (using sine and cosine functions) to simulate irregular road changes.
def road_center_value(world_y):
    # Define base road center based on segments.
    if world_y > 5000:
        base = 8
    elif world_y > 4000:
        t = (5000 - world_y) / 1000.0
        base = 8 + t * 4
    elif world_y > 3000:
        t = (4000 - world_y) / 1000.0
        base = 12 - t * 4
    elif world_y > 2000:
        t = (3000 - world_y) / 1000.0
        base = 8 + t * 4
    else:
        t = (2000 - world_y) / 2000.0
        base = 12 - t * 4

    # The sine and cosine functions are used to generate periodic fluctuations of irregular with amplitudes of 0.8 and 0.5, creating road curvature and undulation.
    if world_y > 5500:
        irregular = 0.0
    else:
        irregular = 0.8 * math.sin(world_y / 100.0) + 0.5 * math.cos(world_y / 50.0)
    return base + irregular

# Calculate the offset according to world_y
def get_track_offset(world_y):
    center_cells = road_center_value(world_y)
    road_center_pixel = center_cells * CELL_SIZE + (CELL_SIZE / 2)
    return road_center_pixel - TRACK_CENTER

# The track gets wider and narrower depending on the height world_y.
def get_road_half_width(world_y):
    if world_y > 5000:
        return WIDE_ROAD_HALF_WIDTH
    elif world_y > 3500:
        return MEDIUM_ROAD_HALF_WIDTH
    else:
        return NARROW_ROAD_HALF_WIDTH


# Drawing Functions
def draw_car(surface, car, screen_x, screen_y):

    # A polygon of 8 points that simulates the shape of a car and is drawn with pygame.draw.polygon().
    body_points = [
        (screen_x, screen_y + car.height * 0.8),
        (screen_x + car.width * 0.2, screen_y + car.height),
        (screen_x + car.width * 0.8, screen_y + car.height),
        (screen_x + car.width, screen_y + car.height * 0.8),
        (screen_x + car.width, screen_y + car.height * 0.4),
        (screen_x + car.width * 0.8, screen_y),
        (screen_x + car.width * 0.2, screen_y),
        (screen_x, screen_y + car.height * 0.4)
    ]
    pygame.draw.polygon(surface, car.color, body_points)

    # Define the polygon of the car window and fill it with the window colour (light blue).
    window_color = (200, 200, 255)
    window_points = [
        (screen_x + car.width * 0.3, screen_y + car.height * 0.1),
        (screen_x + car.width * 0.7, screen_y + car.height * 0.1),
        (screen_x + car.width * 0.65, screen_y + car.height * 0.4),
        (screen_x + car.width * 0.35, screen_y + car.height * 0.4)
    ]
    pygame.draw.polygon(surface, window_color, window_points)

    # Draw two circles with a black outer circle and a grey inner circle to simulate a wheel effect.
    outer_radius = 8
    inner_radius = 4
    wheel_fl = (int(screen_x + car.width * 0.25), int(screen_y + car.height * 0.2))
    wheel_fr = (int(screen_x + car.width * 0.75), int(screen_y + car.height * 0.2))
    for pos in [wheel_fl, wheel_fr]:
        pygame.draw.circle(surface, (0, 0, 0), pos, outer_radius)
        pygame.draw.circle(surface, (192, 192, 192), pos, inner_radius)

    # Drawing the tail flame
    flame_points = [
        (screen_x + car.width * 0.3, screen_y + car.height),
        (screen_x + car.width * 0.5, screen_y + car.height + 20),
        (screen_x + car.width * 0.7, screen_y + car.height)
    ]
    pygame.draw.polygon(surface, (255, 100, 0), flame_points)

    # Display the number on the vehicle
    font = pygame.font.SysFont("Arial", 20)
    text = font.render(str(car.id), True, (255, 255, 255))
    surface.blit(text, (screen_x + car.width / 3, screen_y + car.height / 3))



# Racing Car (Physical Agent)

# Setting car size and speed limits
CAR_WIDTH = 40
CAR_HEIGHT = 80
MIN_SPEED = 3.0
MAX_SPEED = 15.0
START_SPEED = 5.0


class RacingCar:

    # Initialise the car's number, colour, start position, etc. and set the initial speed and dimensions. Define some status information
    def __init__(self, agent_id, color, start_x, start_y):
        self.id = agent_id
        self.color = color
        self.start_x = start_x
        self.start_y = start_y
        self.x = start_x
        self.y = start_y
        self.speed = START_SPEED
        self.width = CAR_WIDTH
        self.height = CAR_HEIGHT
        self.finished = False
        self.crashed = False
        self.total_reward = 0
        self.rank = None
        self.prev_position = self.y

    # Restores the car to its initial state, resetting the position, speed, bonus and status flags.
    def reset(self):
        self.x = self.start_x
        self.y = self.start_y
        self.speed = START_SPEED
        self.finished = False
        self.crashed = False
        self.total_reward = 0
        self.rank = None
        self.prev_position = self.y

    # Updates the state based on the incoming action number.
    # The distance travelled is calculated based on the y-values before and after the update and is returned as an instant reward.
    def update(self, action):
        if action == 1:
            self.speed = min(self.speed + 1.0, MAX_SPEED) # Use min to ensure that MAX_SPEED is not exceeded.
        elif action == 2:
            self.speed = max(self.speed - 1.0, MIN_SPEED) # Use max to ensure that MIN_SPEED is not undered.
        if action == 3:
            self.x -= 10 # Move to left
        elif action == 4:
            self.x += 10 # Move to right
        old_y = self.y
        self.y -= self.speed
        step_distance = old_y - self.y
        # The distance advanced by the step is returned directly as an immediate bonus.
        reward = step_distance
        # The closer we get to the finish line, the greater the reward.
        self.total_reward = LAP_DISTANCE - self.y
        return reward

    # Returns a pygame.Rect object for subsequent collision detection or rendering.
    def get_rect(self):
        return pygame.Rect(int(self.x), int(self.y), int(self.width), int(self.height))


# Multi-Agent Racing Environment
FINISH_LINE = 0

# Construction and Initialisation
class MultiAgentRacingEnv:
    def __init__(self, num_agents):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Two Car Racing: Q-Learning vs DQN")
        self.clock = pygame.time.Clock() # Used to control the game frame rate.
        self.num_agents = num_agents
        self.agents = []
        self.finish_order = []
        self.init_agents()
        self.fixed_screen_y = SCREEN_HEIGHT - CAR_HEIGHT - 10

    # Initialising the Intelligence
    def init_agents(self):
        bottom_center = TRACK_CENTER + get_track_offset(LAP_DISTANCE) # Get the horizontal centre coordinates
        num_agents = self.num_agents
        spread = 40
        for i in range(num_agents):
            offset_i = -spread / 2 + i * (spread / (num_agents - 1)) if num_agents > 1 else 0
            start_x = bottom_center - (CAR_WIDTH / 2) + offset_i
            start_y = LAP_DISTANCE
            # The first intelligent body bit is red and the second intelligent body is blue.
            color = (255, 0, 0) if (num_agents == 2 and i == 0) else (0, 0, 255) if num_agents == 2 else (
            random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
            car = RacingCar(agent_id=i, color=color, start_x=start_x, start_y=start_y)
            self.agents.append(car)

    # Environment reset
    def reset(self):
        self.finish_order = []
        for agent in self.agents:
            agent.reset()
        return self.get_states()

    # Update status and get instant rewards based on actions, add to rewards list.
    def step(self, actions):
        rewards = [0 for _ in range(self.num_agents)]
        old_y_positions = [agent.y for agent in self.agents]
        for idx, agent in enumerate(self.agents):
            if not agent.finished and not agent.crashed:
                r = agent.update(actions[idx])
                rewards[agent.id] += r

        # overtaking bonus， bonus 50 points for overtaking vehicles
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if (old_y_positions[i] > old_y_positions[j] and self.agents[i].y < self.agents[j].y):
                    rewards[self.agents[i].id] += 50
                elif (old_y_positions[j] > old_y_positions[i] and self.agents[j].y < self.agents[i].y):
                    rewards[self.agents[j].id] += 50

        # Update collision and completion conditions
        for agent in self.agents:
            # Calculate the deviation value from the current y-coordinate
            if not agent.finished and not agent.crashed:
                offset = get_track_offset(agent.y)
                dynamic_center = TRACK_CENTER + offset
                half_width = get_road_half_width(agent.y)
                car_center_x = agent.x + agent.width / 2
                # Deviation of more than half the width of the road is considered out of bounds.
                if abs(car_center_x - dynamic_center) > half_width:
                    agent.crashed = True
                    rewards[agent.id] -= 100
                # Cars reach or exceed the finish line and are awarded in order of arrival
                if agent.y <= FINISH_LINE:
                    agent.finished = True
                    if agent.id not in self.finish_order:
                        self.finish_order.append(agent.id)
                        bonus = 100 - 50 * (len(self.finish_order) - 1)
                        rewards[agent.id] += bonus
        # Returns a list of new statuses, a list of rewards, and whether the environment is over or not
        return self.get_states(), rewards, self.is_done()

    # Constructing the state space for each intelligence
    def get_states(self):
        states = []
        for agent in self.agents:
            offset = get_track_offset(agent.y)
            dynamic_center = TRACK_CENTER + offset
            half_width = get_road_half_width(agent.y)
            relative_x = (agent.x + agent.width / 2 - dynamic_center) / half_width # Relative offset of car centre from track centre
            normalized_y = agent.y / LAP_DISTANCE # Normalised value of the current y-coordinate of the car with respect to the total length of the track
            normalized_speed = agent.speed / MAX_SPEED # speed normalisation
            # Converting states to NumPy arrays
            state = np.array([relative_x, normalized_y, normalized_speed], dtype=np.float32)
            states.append(state)
        return states

    # Determine if the environment is finished
    def is_done(self):
        return all(agent.finished or agent.crashed for agent in self.agents)

    # Plotting environment
    def draw(self):
        self.screen.fill((30, 30, 30)) # Clear screen, use dark background
        # Filtering out vehicles that are currently still racing
        active_agents = [agent for agent in self.agents if not agent.finished and not agent.crashed]
        # Criteria for being a ‘leader’
        leader_y = min(agent.y for agent in active_agents) if active_agents else 0
        # Calculate the camera offset
        camera_offset = leader_y - self.fixed_screen_y # Screens are always referenced to the frontrunner

        # Drawing roads based on camera offsets and track parameters
        road_color = (80, 80, 80)
        boundary_color = (255, 255, 255)
        # Iterate over each sample point on the screen and calculate the pixel coordinates of the left and right borders of the road, recording them in the left_points and right_points lists in turn.
        left_points = []
        right_points = []
        step = 20
        for sy in range(0, SCREEN_HEIGHT + step, step):
            world_y = camera_offset + sy
            offset = get_track_offset(world_y)
            dynamic_center = TRACK_CENTER + offset
            half_width = get_road_half_width(world_y)
            left_points.append((dynamic_center - half_width, sy))
            right_points.append((dynamic_center + half_width, sy))
        road_polygon = left_points + right_points[::-1]
        pygame.draw.polygon(self.screen, road_color, road_polygon)
        # Splice the left and right boundary points to form a closed polygon and fill in the road colour.
        # Thicker white lines to depict road boundaries
        pygame.draw.lines(self.screen, boundary_color, False, left_points, 5)
        pygame.draw.lines(self.screen, boundary_color, False, right_points, 5)

        # Drawing the finish line， several small rectangles of alternating colours
        finish_half_width = get_road_half_width(FINISH_LINE)
        finish_left = TRACK_CENTER - finish_half_width
        finish_right = TRACK_CENTER + finish_half_width
        screen_finish_y = FINISH_LINE - camera_offset
        segment_width = 10
        num_segments = int((finish_right - finish_left) // segment_width)
        for i in range(num_segments):
            x_start = finish_left + i * segment_width
            color = (255, 0, 0) if i % 2 == 0 else (255, 255, 255)
            rect = pygame.Rect(x_start, screen_finish_y - 5, segment_width, 10)
            pygame.draw.rect(self.screen, color, rect)

        # Call the draw_car function defined earlier for each smart body to draw the car
        for agent in self.agents:
            screen_x = agent.x
            screen_y = agent.y - camera_offset
            draw_car(self.screen, agent, screen_x, screen_y)

        # The top left corner of the screen shows the order in which the vehicles reach the finish line
        font = pygame.font.SysFont("Arial", 24)
        order_text = "Finish Order: " + ", ".join(str(id) for id in self.finish_order)
        text_surface = font.render(order_text, True, (255, 255, 255))
        self.screen.blit(text_surface, (10, 10))
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        pygame.quit()


# Q-Learning Agent
class QLearner:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=1.0,
                 epsilon_min=0.1, epsilon_decay=0.995):
        self.actions = actions # A list of discrete actions for use in Q-table indexing.
        self.alpha = alpha # alpha: learning rate
        self.gamma = gamma # gamma: discount factor
        self.epsilon = epsilon # epsilon: exploration rate
        self.epsilon_min = epsilon_min # q_table: stores the Q-values corresponding to the state-action in dictionary form.
        self.epsilon_decay = epsilon_decay # set the epsilon decay rate
        self.q_table = {} # Stores the state and the Q-value of each action as a dictionary.
        # Tracks the average change in Q as it is updated, reflecting convergence.
        self.delta_sum = 0.0
        self.num_updates = 0

    # Rounding successive states into tuples.
    # To avoid the state dimension being too fine-grained to affect lookups.
    def discretize_state(self, state):
        return tuple(round(s, 1) for s in state)

    # If the current discretisation state is not in the Q table, it is initialised with the initial value 0 for each action.
    # Use epsilon-greedy strategy: randomly select actions with some probability; otherwise select the action with the largest Q value.
    def choose_action(self, state):
        key = self.discretize_state(state)
        if key not in self.q_table:
            self.q_table[key] = {a: 0.0 for a in self.actions}
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return max(self.q_table[key], key=self.q_table[key].get)

    def learn(self, state, action, reward, next_state, done):
        # Discretise the current and successor states and ensure that these states are initialised in the Q-table.
        key = self.discretize_state(state)
        next_key = self.discretize_state(next_state) if next_state is not None else None
        # Compute the current Q value q_predict and the target Q value q_target.
        if key not in self.q_table:
            self.q_table[key] = {a: 0.0 for a in self.actions}
        if next_key is not None and next_key not in self.q_table:
            self.q_table[next_key] = {a: 0.0 for a in self.actions}
        q_predict = self.q_table[key][action]
        if done or next_state is None:
            q_target = reward
        else:
            q_target = reward + self.gamma * max(self.q_table[next_key].values())
        # Calculate the difference (delta) of the change in Q value during the updating process, cumulative updating
        delta = abs(q_target - q_predict)
        self.delta_sum += delta
        self.num_updates += 1
        self.q_table[key][action] += self.alpha * (q_target - q_predict)
        # Decaying epsilon ensures that the exploration rate is gradually reduced until epsilon_min.
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # Used to clear the cumulative Q update variance before each round
    def reset_delta(self):
        self.delta_sum = 0.0
        self.num_updates = 0

    # For monitoring learning convergence
    def get_avg_delta(self):
        if self.num_updates == 0:
            return 0
        return self.delta_sum / self.num_updates


# DQN Agent Components

# For saving individual state transfer data
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayMemory:
    # Storing Transition Data
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    # Adding Transition Data to Memory
    def push(self, *args):
        self.memory.append(Transition(*args))

    # Randomly sampling a specified amount of transfer data
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    # Used to determine if memory meets batch_size requirements.
    def __len__(self):
        return len(self.memory)

# DQN inherits from nn.Module and represents a feed-forward neural network.
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        # Dimension input_dim mapped to 128 neurons
        self.fc1 = nn.Linear(input_dim, 128)
        # Mapping 128 to 64 dimensions
        self.fc2 = nn.Linear(128, 64)
        # Output layer with the number of neurons as the number of actions output_dim.
        self.fc3 = nn.Linear(64, output_dim)
    # The data is first linearly transformed and passed into the ReLU activation function
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Initialise the DQN agent
class DQNAgent:
    def __init__(self, state_dim, action_dim, device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.policy_net = DQN(state_dim, action_dim).to(device) # Actually used to select the action and Creating the Adam Optimiser
        self.target_net = DQN(state_dim, action_dim).to(device) # Calculate target Q value and Creating the Adam Optimiser
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.memory = ReplayMemory(10000)
        self.epsilon = 1.0 # Setting epsilon parameters
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995 # Used for epsilon-greedy strategies.
        self.gamma = 0.9
        self.batch_size = 64
        self.target_update_interval = 10 #Set discount factor gamma, batch size and target network update interval
        self.steps_done = 0
        self.losses = []

    def select_action(self, state):
        self.steps_done += 1
        # Randomly selecting or using the current policy network to calculate the Q-value
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        # Strategic Network Prediction
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            # Disable gradient calculation
            with torch.no_grad():
                q_vals = self.policy_net(state_tensor)
            # Select the optimal action
            return int(q_vals.argmax().item())

    # Storing state transfers into experience playback memory
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    #
    def optimize_model(self):
        # Check if the experience playback buffer is sufficient
        if len(self.memory) < self.batch_size:
            return None

        # Sampling batch data
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Constructing the batch tensor
        state_batch = torch.FloatTensor(np.vstack(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(np.vstack(batch.next_state)).to(self.device)

        # Handling Termination Status Mask
        non_final_mask = torch.tensor([s is not None for s in batch.next_state],
                                      device=self.device, dtype=torch.bool)

        # Calculate the current Q value
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Calculate the maximum Q value for the next state
        next_state_values = torch.zeros(self.batch_size, 1, device=self.device)
        if non_final_mask.sum().item() > 0:
            next_state_values[non_final_mask] = self.target_net(next_state_batch).max(1)[0].detach().unsqueeze(1)

        # Calculating Desired Q
        expected_values = reward_batch + self.gamma * next_state_values

        # Calculating Desired Q
        loss = nn.MSELoss()(state_action_values, expected_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #  Exploration rate decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Recorded losses
        self.losses.append(loss.item())
        return loss.item()

    # The copy operation copies the policy network parameters to the target network
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())



# Main Function: Training, Validation, and Plotting
def main():
    # Define training epoch and test epoch
    training_episodes = 1000
    validation_episodes = 5
    total_episodes = training_episodes + validation_episodes

    # Performance scores = distance traveled per step.
    performance_scores_q_train = []
    performance_scores_dqn_train = []
    performance_scores_q_val = []
    performance_scores_dqn_val = []

    scores_q_train = []
    scores_dqn_train = []
    q_convergence_train = []
    dqn_loss_train = []

    # Record success (finished=True) per episode.
    success_q_train = []
    success_dqn_train = []

    # Creating an environment with 2 intelligences (racing cars)
    env = MultiAgentRacingEnv(num_agents=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialise the Q-Learning and DQN agents respectively.
    q_agent_rl = QLearner(actions=ACTIONS)
    dqn_agent = DQNAgent(state_dim=3, action_dim=len(ACTIONS), device=device)

    # training epoch
    for episode in range(training_episodes):

        # Initialising environments and intelligences
        states = env.reset()
        q_agent_rl.reset_delta()
        dqn_loss_list = []
        steps = 0
        done = False

        #  Event Handling (Pygame Interaction)
        while not done:
            steps += 1
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # Intelligent Body Movement Options
            state_q = states[0]
            state_dqn = states[1]
            action_q = q_agent_rl.choose_action(state_q)
            action_dqn = dqn_agent.select_action(state_dqn)
            actions = [action_q, action_dqn]

            # Environmental interaction and data collection
            next_states, rewards, done = env.step(actions)

            #  Intelligent Body Learning Process
            q_agent_rl.learn(state_q, action_q, rewards[0], next_states[0], done)
            dqn_agent.store_transition(state_dqn, action_dqn, rewards[1], next_states[1], done)
            loss_val = dqn_agent.optimize_model()
            if loss_val is not None:
                dqn_loss_list.append(loss_val)

            # Status Updates and Rendering
            states = next_states
            env.draw()

        # Calling the DQN-agent method to update the target network, Improve training stability.
        if (episode + 1) % dqn_agent.target_update_interval == 0:
            dqn_agent.update_target_network()

        # Recording of statistical information for the round
        scores_q_train.append(env.agents[0].total_reward)
        scores_dqn_train.append(env.agents[1].total_reward)
        q_convergence_train.append(q_agent_rl.get_avg_delta())
        avg_dqn_loss = np.mean(dqn_loss_list) if dqn_loss_list else 0
        dqn_loss_train.append(avg_dqn_loss)
        perf_q = (LAP_DISTANCE - env.agents[0].y) / (steps + 1e-5)
        perf_dqn = (LAP_DISTANCE - env.agents[1].y) / (steps + 1e-5)
        performance_scores_q_train.append(perf_q)
        performance_scores_dqn_train.append(perf_dqn)
        success_q_train.append(1 if env.agents[0].finished else 0)
        success_dqn_train.append(1 if env.agents[1].finished else 0)

        # Printing the number of rounds and various statistics.
        print(f"[Training] Episode {episode + 1}/{training_episodes} - Q Reward: {env.agents[0].total_reward:.2f}, "
              f"DQN Reward: {env.agents[1].total_reward:.2f}, Avg Δ: {q_agent_rl.get_avg_delta():.4f}, "
              f"DQN Loss: {avg_dqn_loss:.4f}, Performance Q: {perf_q:.2f}, Performance DQN: {perf_dqn:.2f}")

    # Setting the exploration rate of intelligences
    q_agent_rl.epsilon = 0.0
    dqn_agent.epsilon = 0.0

    for episode in range(validation_episodes):

        states = env.reset()
        steps = 0
        done = False
        while not done:
            steps += 1

            # Processing Pygame Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # Intelligent Body Selection Action
            state_q = states[0]
            state_dqn = states[1]
            action_q = q_agent_rl.choose_action(state_q)
            action_dqn = dqn_agent.select_action(state_dqn)
            actions = [action_q, action_dqn]
            next_states, rewards, done = env.step(actions)
            states = next_states
            env.draw()

        # Calculate the score of the intelligences
        score_q = env.agents[0].total_reward
        score_dqn = env.agents[1].total_reward

        # Performance of computational intelligences
        perf_q = (LAP_DISTANCE - env.agents[0].y) / (steps + 1e-5)
        perf_dqn = (LAP_DISTANCE - env.agents[1].y) / (steps + 1e-5)

        # Storage Performance Data
        performance_scores_q_val.append(perf_q)
        performance_scores_dqn_val.append(perf_dqn)
        print(f"[Validation] Episode {episode + 1}/{validation_episodes} - Q Reward: {score_q:.2f}, "
              f"DQN Reward: {score_dqn:.2f}, Performance Q: {perf_q:.2f}, Performance DQN: {perf_dqn:.2f}")

    episodes_train = range(1, training_episodes + 1)
    episodes_val = range(training_episodes + 1, total_episodes + 1)

    # Plot 1: Learning Curve (Cumulative Reward vs. Episode).
    plt.figure(figsize=(16, 10))  # Increased figure size to zoom in on the curves
    plt.plot(episodes_train, scores_q_train, label="Q Reward (Train)", marker='o')
    plt.plot(episodes_train, scores_dqn_train, label="DQN Reward (Train)", marker='s')
    plt.xlabel("Training Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("Learning Curve (Score Curve)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot 2: ROC Curve (Success Rate vs. Episode).
    plt.figure(figsize=(16, 10))
    running_success_q = np.cumsum(success_q_train) / np.arange(1, training_episodes + 1)
    running_success_dqn = np.cumsum(success_dqn_train) / np.arange(1, training_episodes + 1)
    plt.plot(episodes_train, running_success_q, label="Q Success Rate", marker='o')
    plt.plot(episodes_train, running_success_dqn, label="DQN Success Rate", marker='s')
    plt.xlabel("Training Episode")
    plt.ylabel("Success Rate")
    plt.title("ROC Curve (Success Rate)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot 3: Convergence Curve. Avg Δ and DQN loss
    plt.figure(figsize=(16, 10))
    plt.plot(episodes_train, q_convergence_train, label="Q Avg Δ", marker='o')
    plt.plot(episodes_train, dqn_loss_train, label="DQN Avg Loss", marker='s')
    plt.xlabel("Training Episode")
    plt.ylabel("Avg Update Δ / Loss")
    plt.title("Convergence Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot 4: Training Set Performance. Plot the average distance travelled at each step (normalised performance score)
    plt.figure(figsize=(16, 10))
    plt.plot(episodes_train, performance_scores_q_train, label="Q Performance (Train)", marker='o')
    plt.plot(episodes_train, performance_scores_dqn_train, label="DQN Performance (Train)", marker='s')
    plt.xlabel("Training Episode")
    plt.ylabel("Performance Score (Distance/Steps)")
    plt.title("Training Set Performance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot 5: Validation Set Performance.
    plt.figure(figsize=(16, 10))
    plt.plot(episodes_val, performance_scores_q_val, label="Q Performance (Validation)", marker='o')
    plt.plot(episodes_val, performance_scores_dqn_val, label="DQN Performance (Validation)", marker='s')
    plt.xlabel("Validation Episode")
    plt.ylabel("Performance Score (Distance/Steps)")
    plt.title("Validation Set Performance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Demo Phase:
    # Demonstration phase using trained model
    print("Demo phase starting...")
    states = env.reset()
    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        action_q = q_agent_rl.choose_action(states[0])
        action_dqn = dqn_agent.select_action(states[1])
        # Select the action and update the environment
        actions = [action_q, action_dqn]
        states, rewards, done = env.step(actions)
        env.draw()
        pygame.time.delay(30)
    print("Demo finished!")
    pygame.time.delay(3000)
    env.close()


if __name__ == "__main__":
    main()