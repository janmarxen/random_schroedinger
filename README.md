# Random Schrödinger Operator Solver

This repository contains Julia scripts to solve exercises related to the random Schrödinger operator using numerical methods. 

## File Structure

1. **`rand_schroed_script.jl`**  
   Contains the main code to output the solutions for the requested exercises. The execution is divided into two main functions:
   - `main_1D`: Solves the 1D exercises (1 to 6).
   - `main_2D`: Solves the 2D exercises (7 to 10).  
   Running this script will execute both functions sequentially.
   A function for each exercise is declared and run in the main functions. Where possible, exercise results are tested against ground truth using julia state-of-the-art libraries.  

2. **`rand_schroed_methods.jl`**  
   Implements the numerical methods required to solve the exercises.

3. **`rand_schroed_utils.jl`**  
   Provides utility functions used by the main script.

## How to Run

Execute the main script:

```bash
julia rand_schroed_script.jl
