# modules/tests/manual_test_preprocessor_embedder_only.py
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
from modules.intersection.feature_extractor.embedder import (
    LaneSetAttentionEmbedder,
    EmbedderHyperparameters,
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

    # ---- State + Preprocess -------------------------------------------------
    tls_id = "tlJ"
    state_module = StateModule(traci_connection=traci, tls_id=tls_id)

    config = PreprocessorConfig(
        include_queue=True,
        include_approach=True,
        include_total_wait=True,
        include_max_wait=True,
        include_total_speed=True,
    )
    # Adjust args to match your dataclass signature
    norm_params = PreprocessorNormalisationParameters(
        max_detection_range_m=50.0,
        avg_vehicle_occupancy_length_m=7.5,
        max_wait_time_horizon_s=36.0,
    )
    preprocessor = PreprocessorModule(
        traci_connection=traci,
        tls_id=tls_id,
        config=config,
        normalisation_params=norm_params,
    )

    # ---- Embedder (learnable lane set attention) ---------------------------
    # We infer in_dim (= number of columns) after the first state-tensor build.
    embedder: LaneSetAttentionEmbedder | None = None

    step = 0
    while traci.simulation.getMinExpectedNumber() > 0 and step < 100:
        traci.simulationStep()

        # Read & print raw lane-level state
        lanes_state = state_module.read_state()
        print(f"Step {step}:")
        for lane in lanes_state:
            print(
                f"  Lane {lane.lane_id}: queue={lane.queue}, approach={lane.approach}, "
                f"total_wait_s={lane.total_wait_s:.1f}, max_wait_s={lane.max_wait_s:.1f}, "
                f"total_speed={lane.total_speed:.1f}"
            )

        # Build the lane-by-feature tensor (L x F) in the preprocessor's lane order
        X = preprocessor.get_state_tensor(lanes_state)  # torch.Tensor [L, F]
        print(f"  State Tensor (shape {tuple(X.shape)}): {X}")

        # Lazily instantiate the embedder once we know F
        if embedder is None:
            in_dim = int(X.shape[1])
            hyperparams = EmbedderHyperparameters(
                intermediate_vector_length=64,
                post_aggregation_hidden_length=64,
                output_vector_length=64,
                num_seeds=2,
                dropout=0.0,
                layernorm=True,
            )
            embedder = LaneSetAttentionEmbedder(
                num_of_features=in_dim, hyperparams=hyperparams
            )
            print("  Lane order used by Preprocessor:", preprocessor.lane_order)

        assert embedder is not None

        # Forward through embedder: returns (z, weights)
        #  - z: [out_dim]
        #  - weights: [num_seeds, L] (softmax over lanes per seed)
        z, weights = embedder(X)

        # Pretty-print attention per seed (sums to 1 over lanes for each seed)
        for k in range(weights.shape[0]):
            w = weights[k]  # [L]
            w_list = [float(v) for v in w.tolist()]
            pairs = list(zip(preprocessor.lane_order, w_list))
            pairs_str = ", ".join([f"{lid}:{wt:.3f}" for lid, wt in pairs])
            print(f"  Attention seed {k} over lanes -> {pairs_str}")

        print(f"  Lane-set embedding z (shape {tuple(z.shape)}): {z}")

        step += 1

    traci.close()
    sys.stdout.flush()
