using LinearAlgebra
using Dierckx

const Δa = 0.1
const Δδ = 0.1
const MAX_a = 3.5
const MIN_a = -4.0
const MAX_δ = 0.6
const MIN_δ = -0.6
const W = [1.0, 1.0, 1.0, 0.5, 0.01]

"""
	state_cost(s::MPCState; p::Int64=2, weight::Bool=false)
Norm of a state difference vector (weighted if needed).
"""
function state_cost(s::MPCState; p::Int64=2, weight::Bool=false)
	sv = state_vec(s)
	if weight
		return norm(W .* sv, p)
	else
		return norm(sv, p)
	end
end

"""
	traj_cost!(s::MPCState, params::Vector{Float64}, target::MPCState,
					hyperparams::Vector{Float64}; λ::Float64=1.0)
Calculate the cost of the trajectory from initial state `s` given a set of
parameters and a desired final state `target`.
"""
function traj_cost!(s::MPCState, params::Vector{Float64}, target::MPCState,
					hyperparams::Vector{Float64};
					λ::Float64=1.0, p::Int64=2, weight::Bool=false)
	s = generate_last_state!(s, params, hyperparams)
	Δs = state_diff(target, s)
	return state_cost(Δs, p=p, weight=weight) #+ λ*norm(params)
end

function state_diff(target::MPCState, curr::MPCState)
    res = MPCState()
    res.x = target.x - curr.x
    res.y = target.y - curr.y
    res.θ = WrapPosNegPi(target.θ - curr.θ)
    res.v = target.v - curr.v
    res.β = WrapPosNegPi(target.β - curr.β)

    return res
end

function get_jacobian_column(target::MPCState, params::Vector{Float64},
								col::Int64, hyperparams::Vector{Float64},
								initial::MPCState)
	new_params = copy(params)
	degree = Int(length(params) / 2)
	factor = col ≤ degree ? Δa : Δδ

	s = MPCState()
	set_initial_state!(s, initial)

    new_params[col] += factor
    s = generate_last_state!(s, new_params, hyperparams)
    Δs_pos = state_diff(target, s)
    Δs_pos = state_vec(Δs_pos)

	set_initial_state!(s, initial)

    new_params[col] -= 2*factor
    s = generate_last_state!(s, new_params, hyperparams)
    Δs_neg = state_diff(target, s)
    Δs_neg = state_vec(Δs_neg)

    Δs_col = (Δs_pos - Δs_neg) ./ (2.0 * factor)
    return Δs_col
end

"""
	calc_jacobian(target::MPCState, params::Vector{Float64},
						hyperparams::Vector{Float64},
						initial::MPCState)
Calculate the state Jacobian with respect to the trajectory parameters.
"""
function calc_jacobian(target::MPCState, params::Vector{Float64},
						hyperparams::Vector{Float64},
						initial::MPCState)
    J = zeros(initial.dims, length(params))
    for col in 1:length(params)
        J[:, col] = get_jacobian_column(target, params,
											col, hyperparams, initial)
    end
    return J
end

"""
	α_line_search(Δp::Vector{Float64}, params::Vector{Float64},
								target::MPCState, hyperparams::Vector{Float64},
								initial::MPCState)
Step size line search.
"""
function α_line_search(Δp::Vector{Float64}, params::Vector{Float64},
                                target::MPCState, hyperparams::Vector{Float64},
								initial::MPCState;
								p::Int64=2,
								weight::Bool=false)
    mincost = Inf
    s = MPCState()
	set_initial_state!(s, initial)

	best_α = nothing
    for α in 1.0:0.05:2.0
		if α != 0
	        test_params = params .+ (α .* Δp)
	    	c = traj_cost!(s, test_params, target, hyperparams,
	    													p=p, weight=weight)

	        if c < mincost
	            mincost = c
	            best_α = α
	        end
		end
	end

    return best_α
end

function set_initial_state!(s::MPCState, initial::MPCState)
	if !isnothing(initial)
		s.x = initial.x
		s.y = initial.y
		s.θ = initial.θ
		s.v = initial.v
		s.β = initial.β
	else
		s.x = 0.0
		s.y = 0.0
		s.θ = 0.0
		s.v = 0.0
		s.β = 0.0
	end
end

"""
	optimise_trajectory(target::MPCState, params::Vector{Float64},
								hyperparams::Vector{Float64};
								initial::MPCState=nothing,
								iters::Int64=50, min_cost::Float64=0.1,
								early_term::Float64=1e-9)
Optimise MPC trajectory parameters.
"""
function optimise_trajectory(target::MPCState, params::Vector{Float64},
                            	hyperparams::Vector{Float64};
								initial::MPCState=nothing,
								iters::Int64=50, min_cost::Float64=1e-9,
								early_term::Float64=1e-12,
								p::Int64=2,
								weight::Bool=false,)
	a1 = nothing
	δ1 = nothing
	s_fin = nothing
	states = nothing
	old_cost = Inf
	degree = Int(length(params) / 2)
    for i in 1:iters
		s = MPCState()
		set_initial_state!(s, initial)

		s, a1, δ1, states = generate_trajectory!(s, params, hyperparams)
		s_fin = copy(s)
        Δs = state_diff(target, s)
		set_initial_state!(s, initial)
		c = traj_cost!(s, params, target, hyperparams, p=p, weight=weight)
        if c <= min_cost
            break
        end

		if abs(old_cost - c) < early_term
			break
		end
		old_cost = c

        J = calc_jacobian(target, params, hyperparams, initial)
        Δp = copy(params)
        try
            Δp = -pinv(J) * state_vec(Δs)
        catch e
            break
        end

        α = α_line_search(Δp, params, target, hyperparams, initial)
		params += α .* Δp
		params[1:degree] .= clamp!(params[1:degree], MIN_a, MAX_a)
		params[degree+1:end] .= clamp!(params[degree+1:2*degree], MIN_δ, MAX_δ)
    end

    return params, a1, δ1, s_fin, states
end
