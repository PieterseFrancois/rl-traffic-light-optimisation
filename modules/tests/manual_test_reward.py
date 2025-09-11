import os
import sys
import optparse

from sumolib import checkBinary
import traci

from modules.intersection.state import StateModule
from modules.intersection.reward import (
    RewardModule,
    RewardFunction,
    RewardNormalisationParameters,
)

# Import python modules from the $SUMO_HOME/tools directory
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit('Declare environment variable "SUMO_HOME"')


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


# Main entry point
if __name__ == "__main__":
    options = get_options()
    sumoBinary = checkBinary("sumo-gui") if not options.nogui else checkBinary("sumo")
    traci.start(
        [
            sumoBinary,
            "-c",
            "environments/single_intersection/sumo_files/single-intersection-single-lane.sumocfg",
            "--start",
        ]
    )

    # Create StateModule instance
    state_module = StateModule(traci_connection=traci, tls_id="tlJ")
    reward_module = RewardModule(
        traci_connection=traci,
        tls_id="tlJ",
        normalisation_params=RewardNormalisationParameters(
            max_detection_range_m=50.0,
            avg_vehicle_length_m=4.5,
        ),
    )

    reward_module.set_active_reward_function(RewardFunction.QUEUE)

    step = 0
    while traci.simulation.getMinExpectedNumber() > 0 and step < 100:
        traci.simulationStep()
        state = state_module.read_state()
        reward = reward_module.compute_reward(state)

        print(f"Step {step}:")
        for lane in state:
            print(
                f"  Lane {lane.lane_id}: queue={lane.queue}, approach={lane.approach}, total_wait_s={lane.total_wait_s:.1f}, max_wait_s={lane.max_wait_s:.1f}, total_speed={lane.total_speed:.1f}"
            )
        print(f"  Reward: {reward:.3f}")
        step += 1

    traci.close()
    sys.stdout.flush()
