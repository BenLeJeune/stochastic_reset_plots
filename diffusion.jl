module Diffusion
using Distributions


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

"""
The NESS solution to diffusion with resetting to a normal distribution μ, σ^2
"""
function p_st_norm(x, r, D, μ, σ)
    α_0 = sqrt(r/D)
    d = Normal(μ, σ)
    I_1 = exp((μ - x)*α_0)*cdf(d, (x-μ-α_0*(σ^2))/σ)
    I_2 = exp((x - μ)*α_0)*(1 - cdf(d, (x-μ+α_0*(σ^2))/σ))
    return α_0/2*exp((α_0^2*σ^2)/2)*(I_1 + I_2)
end
export p_st_norm


"""
The NESS solution to diffusion with resetting to a uniform distribution on [a, b]
"""
function p_st_unif(x, r, D, a, b)
    α_0 = sqrt(r/D)
    if x < a
        return exp(α_0*x)*(exp(-α_0*a) - exp(-α_0*b)) / (2*(b-a))
    elseif x > b
        return exp(-α_0*x)*(exp(α_0*b) - exp(α_0*a)) / (2*(b-a))
    else
        return (2 - exp(α_0*(a-x)) - exp(α_0*(x-b))) / (2*(b-a))
    
    end
end

"""
THe NESS solution to diffusion with resetting to an exponential distribution
"""
function p_st_exp(x, r, D, λ)
    α_0 = sqrt(r/D)
    if α_0 == λ
        if x < 0
            return λ/4 * exp(λ*x)
        else
            return (λ^2*x/2 + λ/4)*exp(-λ*x)
        end
    else
        if x < 0
            return α_0 * λ * exp(α_0*x) / (2*(λ + α_0))
        else
            return α_0 * λ / (2*(λ - α_0)) * (exp(-α_0*x) - exp(-λ*x)) + α_0 * λ * exp(-λ*x) / (2*(λ + α_0))
        end
    end
end

end