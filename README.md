### 2D Test Problems ###

All 2D methods can be tested by running `main2D`. The `Config` class (defined in `config2D.py`) specifies all of the necessary parameters for a test run. Most importantly, the `problem` parameter defines the test problem/equation we are solving for - `"poisson"` solves the Poisson equation using the test problem from the AMR paper, and `nonlinear` solves the test problem with no forcing term. The `method` parameter specifies the numerical method used to solve the problem. These are all stored under the `methods/` directory. 

The solver can then be run using `solve` in `main2D` and plotted using `plot`. The method `time_compare` can also be used with two sets of parameters to run a timed test between any two sets of parameters. 
