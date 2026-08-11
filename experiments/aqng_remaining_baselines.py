"""Run only the frozen SGD/Adam baselines for paper-scale follow-up cells.

This thin wrapper reuses the validated paper-scale runner unchanged, but narrows
``FULL_METHODS`` to the two Euclidean baselines that were omitted from the
orientation-only scaling/depth blocks. It is intended to pair with the already
completed AQNG physical/random/aligned results at identical seeds and protocol.
"""

import aqng_v2_benchmark as v2
import aqng_validation_gate as gate


if __name__ == "__main__":
    v2.FULL_METHODS = ("SGD", "Adam")
    gate.main()
