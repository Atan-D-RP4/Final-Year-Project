"""Main entry point for the multivariate financial forecasting system."""

import argparse
import sys
from pathlib import Path

from experiments.phase3_zero_shot import Phase3Experiments
from src.utils.config import load_config
from src.utils.logger import setup_logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Multivariate Financial Forecasting with Chronos",
    )

    parser.add_argument(
        "--phase",
        type=str,
        choices=["3", "phase3", "zero-shot"],
        default="3",
        help="Which phase to run (currently only Phase 3 is implemented)",
    )

    parser.add_argument("--config", type=str, help="Path to configuration file")

    parser.add_argument(
        "--fred-api-key",
        type=str,
        help="FRED API key for economic data",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logger("main", level=getattr(__import__("logging"), args.log_level))

    logger.info("Starting Multivariate Financial Forecasting System")
    logger.info(f"Running Phase {args.phase}")

    try:
        # Load configuration
        config_path = Path(args.config) if args.config else None
        config = load_config(config_path)

        # Override FRED API key if provided
        if args.fred_api_key:
            config.data.fred_api_key = args.fred_api_key

        # Run the specified phase
        if args.phase in ["3", "phase3", "zero-shot"]:
            logger.info("Running Phase 3: Baselines & Zero-Shot Experiments")
            experiments = Phase3Experiments(config)
            results = experiments.run_all_experiments()

            if results:
                logger.info("Phase 3 completed successfully")

                # Print summary
                comparison = results.get("comparison", {})
                best_models = comparison.get("best_models", {})

                if best_models:
                    print("\n" + "=" * 50)
                    print("PHASE 3 RESULTS SUMMARY")
                    print("=" * 50)
                    print("\nBest models by metric:")
                    for metric, info in best_models.items():
                        print(
                            f"  {metric:20}: {info['model']:25} ({info['value']:.4f})",
                        )

                    print(f"\nDetailed results saved to: {experiments.results_dir}")
                    print("=" * 50)
            else:
                logger.error("Phase 3 failed")
                sys.exit(1)
        else:
            logger.error(f"Phase {args.phase} not implemented yet")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error running experiments: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
