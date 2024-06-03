module Diffusion

"""
The drift-diffusion propagator in 1D, with diffusion constant `D` and constant drift `μ`.

By default `D=1`, `μ=0`
"""
G_0(x, t, x_0, D=1, μ=0) = 1/(4pi*D*t)^1/2 * exp(-abs(x - x_0 - μt)^2 / 4D*t )

"""
The NESS (Non-Equilibrium Steady State) solution to the drift-diffusion problem with stochastic resetting at rate `r`,
with diffusion constant `D` and constant drift `μ`.
"""
p_st(x, x_0, r, D=1, μ=0) = 1/2 * sqrt(r/D) * exp(-sqrt(r/D)*abs(x - x_0))

"""
The MFPT (Mean First Passage Time) for a target at x_0 in the 1D diffusion problem (without drift). 
"""
T(X_r, r, D=1) = 1/r * (exp(sqrt(r/D)*X_r) - 1)

end