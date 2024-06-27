import numpy as np
from scipy.integrate import solve_ivp

# Define your complex function dy/dt = f(t, y)
def complex_function(t, y):
    tau = 1
    # Example: a complex function
    return 1/tau * np.power(y, 2) * np.exp(-y * (t/tau))

# Define the time span and initial condition
t_span = (0, 10)  # from t=0 to t=10
y0 = [1]  # initial condition

# Use solve_ivp with RK45 (a variant of RK4 with adaptive step size)
sol = solve_ivp(complex_function, t_span, y0, method='RK45', t_eval=np.linspace(0, 10, 1000))

# Extract the results
t = sol.t
y = sol.y[0]

# Plot the results
import matplotlib.pyplot as plt

plt.plot(t, y)
plt.xlabel('Time')
plt.ylabel('y')
plt.title('Numerical Integration using RK45')
plt.show()
