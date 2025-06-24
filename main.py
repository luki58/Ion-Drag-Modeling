# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 14:02:50 2025

@author: Lukas
"""
import time
start_time = time.time()

import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.optimize import curve_fit, fsolve
import os

import ion_drag_force_schwabe2013 as id_sw
import ion_drag_force_khrapak as id_kh
import ion_drag_force_extraction as id_ex
import model_error as me

from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from skopt.plots import plot_gaussian_process

import pandas as pd

def objective(params, gas_type, I, polarity):
    """Objective function to minimize: Sum of model errors"""
    E_multiplier, ne_multiplier, Te_multiplier, z_base = params
    
    # Call model functions
    id_kh.model_khrapak(gas_type, I, polarity, E_multiplier, ne_multiplier, Te_multiplier, z_base)
    id_sw.model_schwabe(gas_type, I, polarity, E_multiplier, ne_multiplier, Te_multiplier, z_base)
    id_ex.solve_fb_equation(gas_type, I, polarity, E_multiplier, ne_multiplier, Te_multiplier, z_base)
    
    # Compute errors
    error_khrapak = me.compute_model_error(gas_type, I, model="Khrapak")
    error_schwabe = me.compute_model_error(gas_type, I, model="Schabe2013")
    
    
    if isinstance(error_khrapak, dict):
        error_khrapak = error_khrapak.get(f"error_{polarity}", 0)
    if isinstance(error_schwabe, dict):
        error_hybrid = error_schwabe.get(f"error_{polarity}", 0)

    return abs(error_hybrid) + abs(error_khrapak)

def save_to_json(gas_type, I, best_values):
    """Save the best values to a JSON file"""
    folder_path = f"machine_learning_results/"
    os.makedirs(folder_path, exist_ok=True)
    json_file = os.path.join(folder_path, f"results_{gas_type}_{I}.json")
    
    with open(json_file, "w") as file:
        json.dump(best_values, file, indent=4, default=lambda x: float(x))

def bayesian_search(gas_type, I, polarity, max_iters):
    """Bayesian optimization to find best (E_multiplier, ne_multiplier, Te_multiplier)."""
    multipliers = {
        ("Argon", 1.5, "neg"): (0.7, 0.65, 1.05, 1.05),
        ("Argon", 1.5, "pos"): (0.7, 0.65, 1.05, 1.05),
        ("Argon", 1.0, "neg"): (0.8, 0.8, 1.0, 1.05),
        ("Argon", 1.0, "pos"): (0.8, 0.8, 1.0, 1.05),
        ("Neon", 1.0, "pos"): (0.9, 0.8, 0.8, 0.95),
        ("Neon", 1.0, "neg"): (0.9, 0.8, 0.8, 0.95),
        ("Neon", 1.5, "pos"): (1.1, 0.85, 0.85, 0.95),
        ("Neon", 1.5, "neg"): (1.1, 0.85, 0.85, 0.95)
    }
    
    E_base, ne_base, Te_base, z_base = multipliers.get((gas_type, I, polarity), (0.9, 0.8, 1.0, 1.0))
    space = [
        Real(E_base * 0.9, E_base * 2.1, name="E_multiplier"),
        Real(ne_base * 0.7, ne_base * 1.3, name="ne_multiplier"),
        Real(Te_base * 0.8, Te_base * 1.2, name="Te_multiplier"),
        Real(z_base * 1.1, z_base * 1.15, name="z_multiplier")
        #Real(z_base * 0.859, z_base * 1.15, name="z_multiplier")
    ]
    
    @use_named_args(space)
    def wrapped_objective(E_multiplier, ne_multiplier, Te_multiplier, z_multiplier):
        return objective([E_multiplier, ne_multiplier, Te_multiplier, z_multiplier], gas_type, I, polarity)
    
    result = gp_minimize(wrapped_objective, space, n_calls=max_iters, acq_func="PI", xi=0.0175, n_jobs=-1)  # or "PI" or "EI")
    
    errors = result.func_vals  # all evaluated f(x)
    min_so_far = [min(errors[:i+1]) for i in range(len(errors))]
    
    plt.figure(dpi=200)
    plt.plot(min_so_far, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Best f(x) found")
    plt.title("Bayesian Optimization Progress")
    plt.grid(True)
    plt.show()
    
    param_names = [dim.name for dim in space]
    param_df = pd.DataFrame(result.x_iters, columns=param_names)
    param_df['objective'] = result.func_vals
    
    for param in param_names:
        plt.figure(dpi=200)
        plt.plot(param_df[param], marker='o')
        plt.xlabel("Iteration")
        plt.ylabel(param)
        plt.title(f"{param} over Optimization Iterations")
        plt.grid(True)
        plt.show()
    
    return result.x, result.fun

gas_types = ["Neon","Argon"] # ,"Argon"Neon"
currents = [1, 1.5] #1 ,1.5
polarities = ["pos","neg"]#"pos", 
best_parameters = {}
start_time = time.time()
# max search iterations
cycles = 100

#%%
# Run Bayesian Optimization for all combinations
for gas_type in gas_types:
    for I in currents:
        best_values = {}
        for polarity in polarities:
            best_params, best_error = bayesian_search(gas_type, I, polarity, cycles)
            best_values[polarity] = {
                "E_multiplier": best_params[0],
                "ne_multiplier": best_params[1],
                "Te_multiplier": best_params[2],
                "z_multiplier": best_params[3],
                "error": best_error
            }
        best_parameters[(gas_type, I)] = best_values
        save_to_json(gas_type, I, best_values)
        print(f"{gas_type}, {I}mA optimized")

# Re-run models with optimized parameters
print("\nRe-running all models with optimized parameters...")
for (gas_type, I), values in best_parameters.items():
    for polarity, params in values.items():
        #print(f"Running models for {gas_type}, {I}, {polarity} with optimized parameters: {params}")
        id_kh.model_khrapak(gas_type, I, polarity, params["E_multiplier"], params["ne_multiplier"], params["Te_multiplier"], params["z_multiplier"])
        id_sw.model_schwabe(gas_type, I, polarity, params["E_multiplier"], params["ne_multiplier"], params["Te_multiplier"], params["z_multiplier"])
        id_ex.solve_fb_equation(gas_type, I, polarity, params["E_multiplier"], params["ne_multiplier"], params["Te_multiplier"], params["z_multiplier"])

print(f"Cycles ran: {cycles}")
print(f"--- {time.time() - start_time:.2f} seconds ---")

