import os
import sys
import optparse

from sumolib import checkBinary
import traci

from modules.intersection.state import StateModule
from modules.intersection.preprocessor import (
    PreprocessorModule,
    PreprocessorConfig,
    PreprocessorNormalisationParameters,
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

    # Create PreprocessorModule instance with desired configuration
    config = PreprocessorConfig(
        include_queue=True,
        include_approach=True,
        include_total_wait=True,
        include_max_wait=True,
        include_total_speed=True,
    )
    norm_params = PreprocessorNormalisationParameters(
        max_detection_range_m=50.0,
        avg_vehicle_occupancy_length_m=7.5,
        max_wait_time_horizon_s=36.0,
    )
    # norm_params = None
    preprocessor_module = PreprocessorModule(
        traci_connection=traci,
        tls_id="tlJ",
        config=config,
        normalisation_params=norm_params,
    )

    step = 0
    while traci.simulation.getMinExpectedNumber() > 0 and step < 100:
        traci.simulationStep()
        state = state_module.read_state()
        state_tensor = preprocessor_module.get_state_tensor(state)

        print(f"Step {step}:")
        for lane in state:
            print(
                f"  Lane {lane.lane_id}: queue={lane.queue}, approach={lane.approach}, total_wait_s={lane.total_wait_s:.1f}, max_wait_s={lane.max_wait_s:.1f}, total_speed={lane.total_speed:.1f}"
            )

        print(f"  State Tensor: {state_tensor}")

        step += 1

    traci.close()
    sys.stdout.flush()
