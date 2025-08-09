import os
import sys
import optparse
import random
import traci
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

# Import python modules from the $SUMO_HOME/tools directory
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit('Declare environment variable "SUMO_HOME"')

from sumolib import checkBinary
import traci


def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option(
        "--nogui",
        action="store_true",
        default=False,
        help="Run the commandline version of sumo",
    )
    options, args = opt_parser.parse_args()
    return options


TRAFFIC_LIGHT_IDS = [
    "J1-4",
    "J1-5",
    "J1-6",
    "J2-4",
    "J2-5",
    "J2-6",
    "J3-4",
    "J3-5",
    "J3-6",
]

# Storage for queue lengths and waiting times
data = {tl: {"queue": [], "wait_time": []} for tl in TRAFFIC_LIGHT_IDS}
# Possible actions: Change phase duration
ACTIONS = [-5, 0, 5]  # Reduce, Keep, Increase phase time
MAX_PHASE_DURATION = 60
MIN_PHASE_DURATION = 5
DEFAULT_PHASE_DURATION = 30

# Store phase durations
phase_durations = {tl: DEFAULT_PHASE_DURATION for tl in TRAFFIC_LIGHT_IDS}


# Define Actor-Critic Networks
class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(
            self.fc3(x), dim=-1
        )  # Probability distribution over actions


class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Single value for state evaluation


# Create networks
input_dim = len(TRAFFIC_LIGHT_IDS) * 2  # Queue + Wait Time for each light
output_dim = len(ACTIONS)  # Actions: -5s, 0, +5s

actor = Actor(input_dim, output_dim)
critic = Critic(input_dim)

actor_optimizer = optim.Adam(actor.parameters(), lr=0.001)
critic_optimizer = optim.Adam(critic.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

GAMMA = 0.9  # Discount factor
EPSILON = 0.1  # Exploration rate


def get_state():
    """Extract state: queue lengths and waiting times"""
    state = []
    for tl in TRAFFIC_LIGHT_IDS:
        queues = traci.trafficlight.getControlledLinks(tl)
        queue_length = sum(
            traci.lane.getLastStepHaltingNumber(lane[0][0]) for lane in queues
        )
        wait_time = sum(traci.lane.getWaitingTime(lane[0][0]) for lane in queues)
        state.extend([queue_length, wait_time])
    return np.array(state, dtype=np.float32)


def choose_action(state):
    """Actor selects an action"""
    state_tensor = torch.tensor(state, dtype=torch.float32)
    action_probs = actor(state_tensor)
    action_index = torch.multinomial(
        action_probs, 1
    ).item()  # Sample from policy distribution
    return ACTIONS[action_index], action_index


def update_actor_critic(state, action_index, reward, next_state):
    """Train the Actor-Critic networks"""
    state_tensor = torch.tensor(state, dtype=torch.float32)
    next_state_tensor = torch.tensor(next_state, dtype=torch.float32)

    # Compute value estimates
    value = critic(state_tensor)
    next_value = critic(next_state_tensor).detach()
    advantage = reward + GAMMA * next_value - value

    # Critic loss (Mean Squared Error)
    critic_loss = advantage.pow(2).mean()

    # Actor loss (Policy Gradient Loss)
    action_probs = actor(state_tensor)
    log_prob = torch.log(action_probs[action_index])
    actor_loss = -log_prob * advantage.detach()

    # Update networks
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()


def run():
    step = 0
    max_steps = 1000  # Simulation limit

    while traci.simulation.getMinExpectedNumber() > 0 and step < max_steps:
        traci.simulationStep()

        # Collect queue length and waiting time for each traffic light
        for tl in TRAFFIC_LIGHT_IDS:

            state = get_state()  # Get current state

            action, action_index = choose_action(
                state
            )  # Choose action (phase adjustment)
            phase_durations[tl] = np.clip(
                phase_durations[tl] + action, MIN_PHASE_DURATION, MAX_PHASE_DURATION
            )
            traci.trafficlight.setPhaseDuration(tl, phase_durations[tl])  # Apply action

            next_state = get_state()  # Get new state
            reward = -sum(state[::2])  # Negative total queue length

            update_actor_critic(
                state, action_index, reward, next_state
            )  # Train networks

            queues = traci.trafficlight.getControlledLinks(tl)
            queue_length = sum(
                traci.lane.getLastStepHaltingNumber(lane[0][0]) for lane in queues
            )

            wait_time = sum(traci.lane.getWaitingTime(lane[0][0]) for lane in queues)
            # wait_time = 0

            # Store data
            data[tl]["queue"].append(queue_length)
            data[tl]["wait_time"].append(wait_time)

        step += 1

    # After the simulation ends, plot the results
    plot_results()


import matplotlib.pyplot as plt


def plot_results():
    """Function to plot queue lengths and waiting times"""
    plt.figure(figsize=(12, 6))

    # Plot Queue Length
    plt.subplot(2, 1, 1)
    for tl in TRAFFIC_LIGHT_IDS:
        plt.plot(data[tl]["queue"], label=f"Queue {tl}")
    plt.title("Queue Length Over Time")
    plt.xlabel("Simulation Step")
    plt.ylabel("Queue Length")
    plt.legend()

    # Plot Waiting Time
    plt.subplot(2, 1, 2)
    for tl in TRAFFIC_LIGHT_IDS:
        plt.plot(data[tl]["wait_time"], label=f"Wait Time {tl}")
    plt.title("Waiting Time Over Time")
    plt.xlabel("Simulation Step")
    plt.ylabel("Waiting Time (s)")
    plt.legend()

    plt.tight_layout()
    plt.show()


# Main entry point
if __name__ == "__main__":
    options = get_options()

    # Start sumo-gui or sumo
    if options.nogui:
        sumoBinary = checkBinary("sumo")
    else:
        sumoBinary = checkBinary("sumo-gui")

    # Start sumo
    traci.start(
        [
            sumoBinary,
            "-c",
            "sumo-files/3x3/grid_3x3.sumocfg",
            "--tripinfo-output",
            "tripinfo.xml",
        ]
    )
    run()

    traci.close()
    sys.stdout.flush()
