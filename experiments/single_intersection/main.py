import os
import sys
import optparse

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


import experiment_1, experiment_2, experiment_3, experiment_4, experiment_5
import experiment_6, experiment_7, experiment_8, experiment_9, experiment_10

# Main entry point
if __name__ == "__main__":
    options = get_options()
    sumoBinary = checkBinary("sumo-gui") if not options.nogui else checkBinary("sumo")
    traci.start(
        [
            sumoBinary,
            "-c",
            "experiments/single_intersection/sumo_files/single-intersection-single-lane.sumocfg",
            "--tripinfo-output",
            "experiments/experiment1.xml",
            "--start",
        ]
    )

    experiment_1.run()
    # experiment_2.run()
    # experiment_3.run()
    # experiment_4.run()
    # experiment_5.run()
    # experiment_6.run()
    # experiment_7.run() 
    # experiment_8.run()
    # experiment_9.run()
    # experiment_10.run()

    traci.close()
    sys.stdout.flush()
