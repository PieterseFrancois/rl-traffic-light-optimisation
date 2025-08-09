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


# TraCI control loop
def run():

    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step += 1
        print("Step: ", step)

        # traci.trafficlight.setRedYellowGreenState('tlCC1', "rryGg")


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
            "sumo-files/test-net/test-net.sumocfg",
            "--tripinfo-output",
            "tripinfo.xml",
        ]
    )
    run()

    traci.close()
    sys.stdout.flush()
