import argparse
from run import run

def main():
    ap = argparse.ArgumentParser(
        description="RLlib independent PPO vs baseline on MultiTLSParallelEnv"
    )
    ap.add_argument("--config-file", type=str, required=True)
    ap.add_argument("--hyperparams-file", type=str, required=True)
    ap.add_argument("--outdir", default="runs/rllib_independent_ppo")
    ap.add_argument(
        "--freeflow-speed-mps",
        type=float,
        default=None,
        help="Optional free-flow speed (m/s) used for TTI computation in KPI export.",
    )
    args = ap.parse_args()

    run(
        config_file=args.config_file,
        hyperparams_file=args.hyperparams_file,
        outdir=args.outdir + "_debug",
        freeflow_speed_mps=args.freeflow_speed_mps,
    )

if __name__ == "__main__":
    main()
