"""Use the installed humanoid_sim package for a finite single-joint smoke.

python examples/single_joint.py --backend cpu --steps 120
python examples/single_joint.py --backend isaac --headless --steps 120
python examples/single_joint.py --backend isaac --visualizer viser --realtime --steps 250000
"""

import sys
from humanoid_sim.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["smoke", *sys.argv[1:]]))
