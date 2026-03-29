"""
Experiment configurations for SCOUT.

Select an experiment by setting the SCOUT_EXPERIMENT environment variable
before running. Valid values: gso, ycb_dcd, ycb_cdl1, ycb_cdl2, ycb_iou,
ycb_emd, ycb_geo, ycb_struct.

Example:
    SCOUT_EXPERIMENT=gso PYTHONPATH=scripts python scripts/run_scripts_gso.py
"""

import os

# Automatically resolve project root (two levels up from this file: utils/ -> scripts/ -> SCOUT_model_routing/)
_folder_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# All experiment configurations
# ---------------------------------------------------------------------------

EXPERIMENTS = {
    'gso': {
        'data_path': os.path.join(_folder_path, 'data', 'gso_data_huny_inst_trel_trip_dcd.npz'),
        'results_folder': os.path.join(_folder_path, 'results', 'gso'),
        'model_categories_path': os.path.join(_folder_path, 'data', 'model_categories_real_gso.csv'),
        'scout_configs': {'T': 1.0, 'beta': 0.7, 'lr': 1e-4, 'epochs': 10, 'alpha': 1e4},
        'scout_no_decouple_configs': {'T': 1.0, 'beta': 0.7, 'lr': 1e-4, 'epochs': 25, 'alpha': 1e4},
    },
    'ycb_dcd': {
        'data_path': os.path.join(_folder_path, 'data', 'ycb_data_huny_inst_trip_trel_dcd.npz'),
        'results_folder': os.path.join(_folder_path, 'results', 'ycb'),
        'model_categories_path': os.path.join(_folder_path, 'data', 'model_categories_real_ycb.csv'),
        'scout_configs': {'T': 0.5, 'beta': 0.5, 'lr': 1e-4, 'epochs': 10, 'alpha': 1e4},
        'scout_no_decouple_configs': {'T': 0.5, 'beta': 0.5, 'lr': 1e-4, 'epochs': 25, 'alpha': 1e4},
    },
    'ycb_cdl2': {
        'data_path': os.path.join(_folder_path, 'data', 'ycb_data_huny_inst_trip_trel_cdl2.npz'),
        'results_folder': os.path.join(_folder_path, 'results', 'ycb'),
        'model_categories_path': os.path.join(_folder_path, 'data', 'model_categories_real_ycb.csv'),
        'scout_configs': {'T': 0.1, 'beta': 0.5, 'lr': 1e-4, 'epochs': 10, 'alpha': 1e4},
        'scout_no_decouple_configs': {'T': 0.1, 'beta': 0.5, 'lr': 1e-4, 'epochs': 25, 'alpha': 1e4},
    },
    'ycb_cdl1': {
        'data_path': os.path.join(_folder_path, 'data', 'ycb_data_huny_inst_trip_trel_cdl1.npz'),
        'results_folder': os.path.join(_folder_path, 'results', 'ycb'),
        'model_categories_path': os.path.join(_folder_path, 'data', 'model_categories_real_ycb.csv'),
        'scout_configs': {'T': 0.2, 'beta': 0.5, 'lr': 1e-4, 'epochs': 10, 'alpha': 1e4},
        'scout_no_decouple_configs': {'T': 0.2, 'beta': 0.5, 'lr': 1e-4, 'epochs': 25, 'alpha': 1e4},
    },
    'ycb_iou': {
        'data_path': os.path.join(_folder_path, 'data', 'ycb_data_huny_inst_trip_trel_iou.npz'),
        'results_folder': os.path.join(_folder_path, 'results', 'ycb'),
        'model_categories_path': os.path.join(_folder_path, 'data', 'model_categories_real_ycb.csv'),
        'scout_configs': {'T': 2.0, 'beta': 0.5, 'lr': 1e-4, 'epochs': 10, 'alpha': 1e4},
        'scout_no_decouple_configs': {'T': 2.0, 'beta': 0.5, 'lr': 1e-4, 'epochs': 25, 'alpha': 1e4},
    },
    'ycb_emd': {
        'data_path': os.path.join(_folder_path, 'data', 'ycb_data_huny_inst_trip_trel_emd.npz'),
        'results_folder': os.path.join(_folder_path, 'results', 'ycb'),
        'model_categories_path': os.path.join(_folder_path, 'data', 'model_categories_real_ycb.csv'),
        'scout_configs': {'T': 0.15, 'beta': 0.5, 'lr': 1e-4, 'epochs': 10, 'alpha': 1e4},
        'scout_no_decouple_configs': {'T': 0.15, 'beta': 0.5, 'lr': 1e-4, 'epochs': 25, 'alpha': 1e4},
    },
    'ycb_geo': {
        'data_path': os.path.join(_folder_path, 'data', 'ycb_data_huny_inst_trip_trel_geo.npz'),
        'results_folder': os.path.join(_folder_path, 'results', 'ycb'),
        'model_categories_path': os.path.join(_folder_path, 'data', 'model_categories_real_ycb.csv'),
        'scout_configs': {'T': 0.1, 'beta': 0.5, 'lr': 1e-4, 'epochs': 10, 'alpha': 1e4},
        'scout_no_decouple_configs': {'T': 0.1, 'beta': 0.5, 'lr': 1e-4, 'epochs': 25, 'alpha': 1e4},
    },
    'ycb_struct': {
        'data_path': os.path.join(_folder_path, 'data', 'ycb_data_huny_inst_trip_trel_struct.npz'),
        'results_folder': os.path.join(_folder_path, 'results', 'ycb'),
        'model_categories_path': os.path.join(_folder_path, 'data', 'model_categories_real_ycb.csv'),
        'scout_configs': {'T': 0.2, 'beta': 0.5, 'lr': 1e-4, 'epochs': 10, 'alpha': 1e4},
        'scout_no_decouple_configs': {'T': 0.2, 'beta': 0.5, 'lr': 1e-4, 'epochs': 25, 'alpha': 1e4},
    },
}

# ---------------------------------------------------------------------------
# Select active experiment from environment variable
# ---------------------------------------------------------------------------

_experiment_name = os.environ.get('SCOUT_EXPERIMENT', 'gso')

if _experiment_name not in EXPERIMENTS:
    raise ValueError(
        f"Unknown SCOUT_EXPERIMENT='{_experiment_name}'. "
        f"Valid options: {', '.join(sorted(EXPERIMENTS.keys()))}"
    )

_cfg = EXPERIMENTS[_experiment_name]

data_path = _cfg['data_path']
results_folder = _cfg['results_folder']
model_categories_path = _cfg['model_categories_path']
scout_configs = _cfg['scout_configs']
scout_no_decouple_configs = _cfg['scout_no_decouple_configs']


START_SEED = 50
END_SEED = 200