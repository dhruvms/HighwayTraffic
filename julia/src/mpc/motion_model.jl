using Parameters

const L_R = 2.0 # Distance between bicycle rear wheel and CoG (m)
const L_F = 2.0 # Distance between bicycle front wheel and CoG (m)

"""
    MPCState
Vehicle state for MPC based on the bicycle model.

# Fields
- `x::Float64 = 0.0` global x position (m)
- `y::Float64 = 0.0` global y position (m)
- `θ::Float64 = 0.0` global heading (rad)
- `v::Float64 = 0.0` velocity (m/s)
- `β::Float64 = 0.0` front wheel steering angle, relative to θ (rad)
"""
@with_kw mutable struct MPCState
	x::Float64 = 0.0
	y::Float64 = 0.0
	θ::Float64 = 0.0 # in radians
	v::Float64 = 0.0
	β::Float64 = 0.0 # in radians

	dims::Int64 = 5
end
Base.copy(s::MPCState) = MPCState(x=s.x, y=s.y, θ=s.θ, v=s.v, β=s.β)

function reset_state!(s::MPCState)
	s.x = 0.0
	s.y = 0.0
	s.θ = 0.0
	s.v = 0.0
	s.β = 0.0
end

function state_vec(s::MPCState)
    v = [s.x, s.y, s.θ, s.v, s.β]
    return v
end

"""
    update_mpc_state!(s::MPCState, a::Float64, δf::Float64, Δt::Float64)
Apply control based on bicycle model.
"""
function update_mpc_state!(s::MPCState, a::Float64, δf::Float64, Δt::Float64)
	s.x += s.v * cos(s.θ + s.β) * Δt
	s.y += s.v * sin(s.θ + s.β) * Δt
	s.θ += (s.v / L_R) * sin(s.β) * Δt
	s.θ = WrapPosNegPi(s.θ)

	s.v += a * Δt
	s.v = max(s.v, 0.0)
	s.v = min(s.v, 15.0)
	s.β = atan( (L_R / (L_F + L_R)) * tan(δf * Δt))
	s.β = WrapPosNegPi(s.β)
	s.β = clamp(s.β, -π/4, π/4)
end

"""
	generate_trajectory!(s::MPCState, params::Vector{Float64},
								hyperparams::Vector{Float64};)
Given a set of parameters that define the acceleration and steering polynomials
over time, generate a bicycle model trajectory from the initial state for a
duration specified by the hyperparameters.
"""
function generate_trajectory!(s::MPCState, params::Vector{Float64},
								hyperparams::Vector{Float64};)
	degree = Int(length(params) / 2)
	n, timestep, interp = hyperparams
	T = timestep * n
	t_knots = collect(range(0, length=degree, stop=T))
	a_knots = params[1:degree]
	δ_knots = params[degree+1:end]

	a_spline = Spline1D(t_knots, a_knots; k=Int64(interp))
	δ_spline = Spline1D(t_knots, δ_knots; k=Int64(interp))

	times = collect(range(0.0, step=timestep, length=Int64(n)+1))

	a_interm = a_spline(times)
	δ_interm = δ_spline(times)

	states = [copy(s)]
    for i in 1:Base.length(times)-1
		update_mpc_state!(s, a_interm[i], δ_interm[i], timestep)
		push!(states, copy(s))
	end

	return s, a_interm[1], δ_interm[1], states
end

"""
	generate_last_state!(s::MPCState, params::Vector{Float64},
								hyperparams::Vector{Float64};)
Return the last state of the generated trajectory.
"""
function generate_last_state!(s::MPCState, params::Vector{Float64},
								hyperparams::Vector{Float64};)
	last, _, _ = generate_trajectory!(s, params, hyperparams::Vector{Float64},)
	return last
end
