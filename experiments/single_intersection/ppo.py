import os
import sys
import traci
import random
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from stable_baselines3 import PPO
from modules import lane_metrics, plotting
from modules.metrics_structs import SingleIntersectionMetrics
from sumo_traffic_env import SumoTrafficEnv  # Your gym.Env wrapper

def run():
    MODEL_PATH = "experiments/single_intersection/ppo_traffic_model_single_intersection"
    model = PPO.load(MODEL_PATH)
    env = SumoTrafficEnv()
    obs, _ = env.reset()

    data = SingleIntersectionMetrics()
    step_count = 0
    phase_changes = 0
    last_phase = -1

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated


        if done:
            break

        step_count += 1
        data.steps.append(step_count)

        # Metrics collection
        waits = lane_metrics.get_lane_average_waiting_times(env.lane_ids)
        speeds = lane_metrics.get_lane_speeds(env.lane_ids)
        queues = lane_metrics.get_lane_queue_lengths(env.lane_ids)

        for dir, lane_id in env.lanes.items():
            data.wait[dir].append(waits[lane_id])
            data.speed[dir].append(speeds[lane_id])
            data.queue[dir].append(queues[lane_id])

        # Track phase changes
        if action != last_phase:
            phase_changes += 1
            last_phase = action
        data.phases.append(action)

    print(f"Total Phase Changes: {phase_changes}")

    # Calculate average metrics
    avg_wait = np.mean([np.mean(data.wait[dir]) for dir in env.lanes])
    avg_speed = np.mean([np.mean(data.speed[dir]) for dir in env.lanes])
    avg_queue = np.mean([np.mean(data.queue[dir]) for dir in env.lanes])

    print(f"Average Wait Time: {avg_wait:.2f} seconds")
    print(f"Average Speed: {avg_speed:.2f} m/s")
    print(f"Average Queue Length: {avg_queue:.2f} vehicles")

    # Plotting
    plotting.init_plot_layout(num_subplots=3)
    plotting.plot_lane_metrics(data.steps, data.wait, data.phases, "PPO Light: Wait Time", "Seconds", [0], [1])
    plotting.plot_lane_metrics(data.steps, data.speed, data.phases, "PPO Light: Speed", "m/s", [0], [1])
    plotting.plot_lane_metrics(data.steps, data.queue, data.phases, "PPO Light: Queue", "Vehicles", [0], [1])
    plotting.show_plots()


if __name__ == "__main__":
    run()
    traci.close()