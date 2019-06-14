"""
Mod(x::T, y::T) and  Wrap*(angle::T) are a bunch of modulus helper functions.
From https://stackoverflow.com/questions/4633177/c-how-to-wrap-a-float-to-the-interval-pi-pi
"""
function Mod(x::T, y::T) where T <: Real
	if y == 0.0
		return x
	end

	m::Float64 = x - y*floor(x/y)

	if y > 0
		if m >= y
			return 0
		end

		if m < 0
			if y + m == y
				return 0
			else
				return y + m
			end
		end
	else
		if m <= y
			return 0
		end

		if m > 0
			if y + m == y
				return 0
			else
				return y + m
			end
		end
	end

	return m
end

function WrapPosNegPi(angle::T) where T <: Real
	return Mod(angle + π, 2π) - π
end

function Wrap2Pi(angle::T) where T <: Real
	return Mod(angle, 2π)
end

function WrapPosNeg180(angle::T) where T <: Real
	return Mod(angle + 180.0, 360.0) - 180.0
end

function Wrap360(angle::T) where T <: Real
	return Mod(angle, 360.0)
end

"""
	GetΔtVec(log::Bool, base::Float64, ord::Float64,
					n::Int64, timestep::Float64)
Sampling function for timesteps (can sample on linear or log scale given a base)
"""
function GetΔtVec(log::Bool, base::Float64, ord::Float64,
					n::Int64, timestep::Float64)
	times = nothing
	if log
		times = (base .^(range(0, stop=ord, length=n)))
		times = (times .- 1.0) ./ (base^ord - 1.0)
		times *= n * timestep
	else
		times = collect(range(0.0, step=timestep, length=n))
	end
	return times
end

"""
	OrnsteinUhlenbeckNoise

Ornstein-Uhlenbeck noise model.
https://en.wikipedia.org/wiki/Ornstein–Uhlenbeck_process
Implementation taken from OpenAI Baselines implementation of DDPG.
https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
"""
mutable struct OrnsteinUhlenbeckNoise
	μ::Vector{Float64}
	σ::Float64
	θ::Float64
	Δt::Float64
	x0::Union{Vector{Float64}, Nothing}
	x_prev::Vector{Float64}

	function OrnsteinUhlenbeckNoise(
		μ::Vector{Float64},
		σ::Float64;
		θ::Float64=0.15,
		Δt::Float64=1e-2,
		x0::Union{Vector{Float64}, Nothing}=nothing
		)
		obj = new()
		obj.μ = μ
		obj.σ = σ
		obj.θ = θ
		obj.Δt = Δt
		obj.x0 = x0
		if isnothing(obj.x0)
			obj.x_prev = zeros(size(obj.μ))
		else
			obj.x_prev = copy(obj.x0)
		end
		obj
	end
end

function OrnsteinUhlenbeckNoise!(noise::OrnsteinUhlenbeckNoise)
	x = noise.x_prev + noise.θ * (noise.μ .- noise.x_prev) * noise.Δt + noise.σ * sqrt(noise.Δt) * randn(size(noise.μ))
	noise.x_prev = x
	return x
end

function OrnsteinUhlenbeckNoise!(noise::OrnsteinUhlenbeckNoise, Δt::Float64)
	x = noise.x_prev + noise.θ * (noise.μ .- noise.x_prev) * Δt + noise.σ * sqrt(Δt) * randn(size(noise.μ))
	noise.x_prev = x
	return x
end

function ResetOrnsteinUhlenbeckNoise!(noise::OrnsteinUhlenbeckNoise)
	if isnothing(noise.x0)
		noise.x_prev = zeros(size(noise.μ))
	else
		noise.x_prev = copy(noise.x0)
	end
end
