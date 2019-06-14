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
								hyperparams::Vector{Float64};
								noisy::Bool=false)
Given a set of parameters that define the acceleration and steering polynomials
over time, generate a bicycle model trajectory from the initial state for a
duration specified by the hyperparameters.
"""
function generate_trajectory!(s::MPCState, params::Vector{Float64},
								hyperparams::Vector{Float64};
								noisy::Bool=false)
	n, timestep, interp = hyperparams
	time = timestep * n
	t_knots = [0.0, time/2.0, time]
	a_knots = params[1:3]
	δ_knots = params[4:6]

	a_spline = Spline1D(t_knots, a_knots; k=Int64(interp))
	δ_spline = Spline1D(t_knots, δ_knots; k=Int64(interp))

	times = collect(range(0.0, step=timestep, length=Int64(n)+1))

	a_interm = a_spline(times)
	δ_interm = δ_spline(times)

	noise_a = OrnsteinUhlenbeckNoise([0.0], 0.2, θ=1.0)
	noise_δ = OrnsteinUhlenbeckNoise([0.0], 0.2, θ=1.0)

	for i in 1:length(times)-1
		if noisy
			a_noise = OrnsteinUhlenbeckNoise!(noise_a, timestep)
			δ_noise = OrnsteinUhlenbeckNoise!(noise_δ, timestep)
			update_mpc_state!(s, a_interm[i] + a_noise[1], δ_interm[i] + δ_noise[1], timestep)
		else
			update_mpc_state!(s, a_interm[i], δ_interm[i], timestep)
		end
	end

	return s, a_interm[1], δ_interm[1]
end

"""
	generate_last_state!(s::MPCState, params::Vector{Float64},
								hyperparams::Vector{Float64};
								noisy::Bool=false)
Return the last state of the generated trajectory.
"""
function generate_last_state!(s::MPCState, params::Vector{Float64},
								hyperparams::Vector{Float64};
								noisy::Bool=false)
	last, _, _ = generate_trajectory!(s, params, hyperparams::Vector{Float64},
									noisy=noisy)
	return last
end
