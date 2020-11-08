**Example Output for gplearn**

The Python package `gplearn` can be used for symbolic regression, and it predicts the relationship between independent and dependent variables.  To perform regression,
use the following:
```
from gplearn.genetic import SymbolicRegressor

est_gp = SymbolicRegressor( insert_parameter values)
est_gp.fit(indep_variable, dep_variable)
```
Please see *integer_seqn_methods.py* in this repository for a more detailed implementation of the symbolic regression on a specific example.  To access the predicted relationship, 
in the command line, type
```
print(est_gp._program)
```
And the output looks similar to
```
add(div(div(log(0.691), div(X0, X0)), div(neg(X0), mul(X0, add(sin(X0), sub(X0, -0.513))))), abs(X0))
```
This is the exact output for one of the integer sequences referenced in *integer_seqn_methods.py* and a forthcoming report of findings.
