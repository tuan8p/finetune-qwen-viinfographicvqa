"""Entry point for mixed single+multi-image inference: python -m src.inference.single_and_multi"""

import sys

from src.inference.single_and_multi.run_inference import main


if __name__ == "__main__":
    sys.exit(main())
