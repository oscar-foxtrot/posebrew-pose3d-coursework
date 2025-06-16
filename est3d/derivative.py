import sympy as sp

# Define the symbol
x = sp.Symbol('x')

# Define the function g(x)
numerator = sp.exp(-0.5 * ((x - 122)/40)**2)
denominator = (
    sp.exp(-0.5 * ((x - 122)/40)**2) +
    sp.exp(-0.5 * ((x - 81 - 122)/40)**2) +
    sp.exp(-0.5 * ((x + 81 - 122)/40)**2)
)
g = numerator / denominator

# Compute the derivative
g_derivative = sp.diff(g, x)
print(sp.simplify(g_derivative, rational=False))