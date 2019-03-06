"""Configuration of experiment and schema"""
from schema import dataset, model, error_type, clean_method, scenario

# =============================================================================
# Directory Configuration
# =============================================================================
data_dir = 'data' # dir storing data
result_dir = 'result' # dir saving experiment results
analysis_dir = 'analysis' # dir saving analysis results
plot_dir = 'plot' # dir saving plots

# =============================================================================
# Experiment Configuration
# =============================================================================
root_seed = 1 # root seed for entire experiments
n_resplit = 20 # num of resplit for handling split randomness
n_retrain = 5 # num of retrain for handling random search randomness
test_ratio = 0.3 # train/test ratio
max_size = 15000 # max data size for training

# =============================================================================
# Schema Configuration
# =============================================================================
datasets = dataset.datasets
models = model.models
error_types = error_type.error_types
scenarios = scenario.scenarios