import os
import sys
import optparse

from sumolib import checkBinary
import traci

from modules.intersection.action import ActionModule, TLSTimingStandards

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

    # Create ActionModule instance
    timing_standards = TLSTimingStandards(yellow_s=3.0, all_red_s=2.0, max_green_s=30.0)

    # inferred_action_module = ActionModule(traci_connection=traci, tls_id="tlJ", timing_standards=timing_standards)
    # inferred_action_module.set_phase(0)  # Set to first green phase
    # traci.simulationStep()

    # # Create ActionModule instance with custom green phases
    # custom_phases = ["GggrrrGGgrrr", "rGgrrrGGgrrr", "GGggggGGgggg", "GGgrrrGGgrrr"]
    # custom_action_module = ActionModule(traci_connection=traci, tls_id="tlJ", green_phase_strings=custom_phases)
    # custom_action_module.set_phase_immediately(2)  # Set to third custom green phase
    # traci.simulationStep()

    timing = TLSTimingStandards(
        min_green_s=5.0, yellow_s=3.0, all_red_s=2.0, max_green_s=12.0
    )
    am = ActionModule(traci_connection=traci, tls_id="tlJ", timing_standards=timing)

    assert am.n_actions >= 2, f"Need >=2 green phases; got {am.n_actions}"
    i0, i1 = 0, 1  # only two phases

    def advance(seconds: float):
        target = float(traci.simulation.getTime()) + seconds
        while float(traci.simulation.getTime()) < target:
            traci.simulationStep()
            am.update_transition()

    def dump(tag: str):
        t = float(traci.simulation.getTime())
        st = traci.trafficlight.getRedYellowGreenState(am.tls_id)
        cur = am.active_phase_memory.phase_index if am.active_phase_memory else None
        print(
            f"[{t:7.1f}] {tag:<20} cur={cur} state={st} in_trans={am.in_transition} "
            f"ready={am.ready_for_decision()}"
        )

    # 1) Set to phase 0 (immediate)
    am.set_phase(i0)
    dump("set 0")
    advance(1.0)

    # 2) Request switch to phase 1 -> should go yellow -> all-red -> commit
    advance(5.0)  # wait a bit for min-green
    am.set_phase(i1)
    dump("request 1 (start)")
    print("mask (during transition):", am.get_action_mask())  # expect only queued True
    advance(timing.yellow_s + timing.all_red_s + 0.3)
    dump("after commit to 1")

    # 3) Min-green block: immediate request back to 0 should be ignored until 5s elapsed
    am.set_phase(i0)  # ignored due to min_green
    dump("req 0 (min-green)")
    advance(5.2)  # satisfy min_green
    am.set_phase(i0)  # now accepted
    dump("req 0 (accepted)")
    advance(timing.yellow_s + timing.all_red_s + 0.3)
    dump("after commit to 0")

    # 4) Max-green mask: wait beyond max, then current phase should be masked out
    advance(timing.max_green_s + 0.2)
    print(
        "mask (max-green overrun):", am.get_action_mask()
    )  # expect current index False
    dump("post max-green")

    traci.close()
    sys.stdout.flush()
