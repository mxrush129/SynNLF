import timeit
import torch
import numpy as np
import pandas as pd
from learn.cegis_lyapunov import Cegis
from utils.Config_V import CegisConfig
from benchmarks.Exampler_V import get_example_by_name
import random
import io
import sys

from learn.cegis_lyapunov import Cegis
from utils.Config_V import CegisConfig
import timeit
import torch
import numpy as np
from benchmarks.Exampler_V import get_example_by_name


# Random seed list (50 seeds)
seeds = random.sample(range(1000, 10000), 50)  # Generate 50 unique seeds

# Different activation functions and hidden layer configurations
activations = ['SKIP', 'MUL', 'SQUARE']
hidden_neurons_options = [5, 10, 20,50,100]

# Store results in a list
results = []

# Track statistics
total_sos_passed = 0
total_t_learn = 0
total_t_cex = 0
total_t_sos = 0
min_t_learn = float('inf')
min_t_cex = float('inf')
min_t_sos = float('inf')

# Loop over all combinations of seeds, activations, and hidden_neurons
for seed in seeds:
    for activation in activations:
        for hidden_neurons in hidden_neurons_options:
            # Set the random seed for reproducibility
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            example = get_example_by_name('C6')  # Example, can be changed to any benchmark
            example.D_zones.r = pow(100, 2)
            # Configurations
            opts = {
                "ACTIVATION": [activation],
                "EXAMPLE": example,
                "N_HIDDEN_NEURONS": [hidden_neurons] * len([activation]),
                "BATCH_SIZE": 500,
                "LEARNING_RATE": 0.001,
                "LOSS_WEIGHT": (1.0, 1.0),
                "SPLIT_D": False,
                'BIAS': False,
                'DEG': [2, 2, 0],
                'max_iter': 20,
                'counter_nums': 50,
                'ellipsoid': True,
                'x0': [5] * example.n,
                'loss_optimization': False,
            }

            Config = CegisConfig(**opts)
            c = Cegis(Config)

            # Create a StringIO stream to capture print statements
            captured_output = io.StringIO()
            sys.stdout = captured_output  # Redirect standard output to the StringIO stream

            # Start timing the execution and get the time values
            learning_time, cex_time, sos_time = c.solve()

            # Now we need to count how many times "SOS verification passed!" was printed
            captured_output.seek(0)  # Go to the beginning of the StringIO stream
            sos_passed_count = captured_output.getvalue().count("SOS verification passed!")

            # Update the SOS verification passed counter
            total_sos_passed += sos_passed_count

            # Update statistics
            total_t_learn += learning_time
            total_t_cex += cex_time
            total_t_sos += sos_time

            min_t_learn = min(min_t_learn, learning_time)
            min_t_cex = min(min_t_cex, cex_time)
            min_t_sos = min(min_t_sos, sos_time)

            # Store the result
            results.append({
                'Seed': seed,
                'Activation': activation,
                'Hidden Neurons': hidden_neurons,
                'Learning Time (seconds)': learning_time,
                'Counter-Example Generation Time (seconds)': cex_time,
                'SOS Verification Time (seconds)': sos_time,
                'SOS Passed Count': sos_passed_count  # Add the SOS passed count
            })

            # Reset sys.stdout
            sys.stdout = sys.__stdout__

# Compute averages
total_tests = len(seeds) * len(activations) * len(hidden_neurons_options)
avg_t_learn = total_t_learn / total_tests
avg_t_cex = total_t_cex / total_tests
avg_t_sos = total_t_sos / total_tests

# Convert results to a pandas DataFrame
df_results = pd.DataFrame(results)

# Add summary statistics to the results
summary = {
    'SOS Verification Passed': total_sos_passed,
    'Average Learning Time (seconds)': avg_t_learn,
    'Average Counter-Example Generation Time (seconds)': avg_t_cex,
    'Average SOS Verification Time (seconds)': avg_t_sos,
    'Min Learning Time (seconds)': min_t_learn,
    'Min Counter-Example Generation Time (seconds)': min_t_cex,
    'Min SOS Verification Time (seconds)': min_t_sos
}

# Save the results to a CSV file
df_results.to_csv('1122experiment_results_with_summary.csv', index=False)

# Save summary statistics in a separate CSV file
df_summary = pd.DataFrame([summary])
df_summary.to_csv('1122experiment_summary.csv', index=False)

print(
    'Experiments complete! Results saved to "experiment_results_with_summary.csv" and summary to "experiment_summary.csv"')

