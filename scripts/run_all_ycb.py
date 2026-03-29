"""Run all YCB experiments across all metrics."""

import os

YCB_METRICS = ['dcd', 'cdl1', 'cdl2', 'iou', 'emd', 'geo', 'struct']


if __name__ == "__main__":
    for metric in YCB_METRICS:
        print(f"\n{'=' * 60}")
        print(f"Starting YCB {metric}")
        print('=' * 60)

        os.system(f"PYTHONPATH=scripts python scripts/ycb_scripts/run_scripts_ycb_{metric}.py")

        print(f"Finished YCB {metric}")
