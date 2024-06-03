module RunAndTumble

"""
A helpful constant within run-and-tumble dynamics.
"""
λ_r = (r, γ, v_0) -> sqrt(r*(r+2γ)/v_0^2)

"""
The steady-state distribution for a run-and-tumble particle under stochastic resetting, with
tumbling rate `γ` and velocity `v_0` with poissonian resetting rate `r`.
"""
p_st(x, r, γ, v_0) = λ_r(r, γ, v_0)/2 * exp(-λ_r(r, γ, v_0)*abs(x))

"""
The MFPT for a run-and-tumble particle that is reset to X_r.
"""
T(X_r, r, γ, v_0) = -1/r + 2γ/r*(exp(λ_r(r, γ, v_0)*X_r)/(r + 2γ - sqrt(r*(r+2γ))))

end