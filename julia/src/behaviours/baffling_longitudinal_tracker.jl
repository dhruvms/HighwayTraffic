"""
	BafflingLongitudinalTracker <: LaneFollowingDriver

Extended Intelligent Driver Model. A rule based driving model that is governed by parameter
settings. The output is an longitudinal acceleration. The desired velocity, however, is purturbed within an offset range for practicality.

# Fields
- `a::Float64 = NaN` the predicted acceleration i.e. the output of the model
- `σ::Float64 = NaN` allows errorable IDM, optional stdev on top of the model, set to zero or NaN for deterministic behavior
- `k_spd::Float64 = 1.0` proportional constant for speed tracking when in freeflow [s⁻¹]
- `δ::Float64 = 4.0` acceleration exponent
- `T::Float64  = 1.5` desired time headway [s]
- `v_des::Float64 = 29.0` desired speed [m/s]
- `s_min::Float64 = 5.0` minimum acceptable gap [m]
- `a_max::Float64 = 3.0` maximum acceleration ability [m/s²]
- `d_cmf::Float64 = 2.0` comfortable deceleration [m/s²] (positive)
- `d_max::Float64 = 9.0` maximum deceleration [m/s²] (positive)
- `v_offset:Float64 = 5.0` offset from the desired speed [m/s]
"""
@with_kw mutable struct BafflingLongitudinalTracker <: LaneFollowingDriver
    a::Float64 = NaN # predicted acceleration
    σ::Float64 = NaN # optional stdev on top of the model, set to zero or NaN for deterministic behavior

    k_spd::Float64 = 1.0 # proportional constant for speed tracking when in freeflow [s⁻¹]

    δ::Float64 = 4.0 # acceleration exponent [-]
    T::Float64  = 1.5 # desired time headway [s]
    v_des::Float64 = 29.0 # desired speed [m/s]
    s_min::Float64 = 2.0 # minimum acceptable gap [m]
    a_max::Float64 = 3.0 # maximum acceleration ability [m/s²]
    d_cmf::Float64 = 2.0 # comfortable deceleration [m/s²] (positive)
    d_max::Float64 = 9.0 # maximum deceleration [m/s²] (positive)
    ΔT::Float64 = 0.2 # timestep to simulate [s]

    v_offset::Float64 = 5.0 # offset from the desired speed [m/s]
end
get_name(::BafflingLongitudinalTracker) = "BafflingLongitudinalTracker"
function AutomotiveDrivingModels.set_desired_speed!(model::BafflingLongitudinalTracker, v_des::Float64)
    model.v_des = v_des
    model
end

function AutomotiveDrivingModels.track_longitudinal!(driver::BafflingLongitudinalTracker, scene::Scene, roadway::Roadway, vehicle_index::Int, fore::NeighborLongitudinalResult)
    v_ego = scene[vehicle_index].state.v
    if fore.ind != nothing
        headway, v_oth = fore.Δs, scene[fore.ind].state.v
    else
        headway, v_oth = NaN, NaN
    end
    return track_longitudinal!(driver, v_ego, v_oth, headway)
end

function AutomotiveDrivingModels.track_longitudinal!(model::BafflingLongitudinalTracker, v_ego::Float64, v_oth::Float64, headway::Float64)
    if !isnan(v_oth)
        @assert !isnan(headway)
        if headway < 0
            @debug("BafflingLongitudinalTracker Warning: BDM received a negative headway $headway"*
                  ", a collision may have occured.")
		    # model.a = -model.d_max
        else
            Δv = v_oth - v_ego
            s_des = model.s_min + v_ego*model.T - v_ego*Δv / (2*sqrt(model.a_max*model.d_cmf))
            v_ratio = model.v_des > 0.0 ? (v_ego/model.v_des) : 1.0
            model.a = model.a_max * (1.0 - v_ratio^model.δ - (s_des/headway)^2)
        end
		if v_ego + model.ΔT * model.a < 0
			model.a = max(-model.d_max, -v_ego/model.ΔT)
		end
    else
        # no lead vehicle, just drive to match desired speed
		v_offset = model.v_des/3
		# Δv = (model.v_des + v_offset*(rand()-0.5))- v_ego
        Δv = (model.v_des)- v_ego
        model.a = Δv*model.k_spd # predicted accel to match target speed
    end

    @assert !isnan(model.a)

    model.a = clamp(model.a, -model.d_max, model.a_max)

    return model
end
function Base.rand(model::BafflingLongitudinalTracker)
    if isnan(model.σ) || model.σ ≤ 0.0
        LaneFollowingAccel(model.a)
    else
        LaneFollowingAccel(rand(Normal(model.a, model.σ)))
    end
end
function Distributions.pdf(model::BafflingLongitudinalTracker, a::LaneFollowingAccel)
    if isnan(model.σ) || model.σ ≤ 0.0
        Inf
    else
        pdf(Normal(model.a, model.σ), a.a)
    end
end
function Distributions.logpdf(model::BafflingLongitudinalTracker, a::LaneFollowingAccel)
    if isnan(model.σ) || model.σ ≤ 0.0
        Inf
    else
        logpdf(Normal(model.a, model.σ), a.a)
    end
end
