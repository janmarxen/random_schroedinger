# Random Schrödinger Operator Solver

This repository contains Julia scripts to solve exercises related to the random Schrödinger operator using numerical methods. 

## File Structure

1. **`rand_schroed_script.jl`**  
   Contains the main code to output the solutions for the requested exercises. The execution is divided into two main functions:
   - `main_1D`: Solves the 1D exercises.
   - `main_2D`: Solves the 2D exercises.  
   Running this script will execute both functions sequentially.

2. **`rand_schroed_methods.jl`**  
   Implements the numerical methods required to solve the exercises.

3. **`rand_schroed_utils.jl`**  
   Provides utility functions used by the main script.

## How to Run

Execute the main script:

```bash
julia rand_schroed_script.jl
