
# abstract type LateralDriverModel end
# get_name(::LateralDriverModel) = "???"
# reset_hidden_state!(model::LateralDriverModel) = model # do nothing by default
# observe!(model::LateralDriverModel, scene::Scene, roadway::Roadway, egoid::Int) = model  # do nothing by default
# Base.rand(model::LateralDriverModel) = error("rand not implemented for model $model")
# Distributions.pdf(model::LateralDriverModel, a_lon::Float64) = error("pdf not implemented for model $model")
# Distributions.logpdf(model::LateralDriverModel, a_lon::Float64) = error("logpdf not implemented for model $model")

# TODO: Why does this struct live in the `lateral_driver_models.jl` file
"""
	BafflingLateralTracker

A controller that executes the lane change decision made by the `lane change models`

# Constructors
	BafflingLateralTracker(;σ::Float64 = NaN,kp::Float64 = 3.0,kd::Float64 = 2.0)

# Fields
- `a::Float64 = NaN` predicted acceleration
- `σ::Float64 = NaN` optional stdev on top of the model, set to zero or NaN for deterministic behavior
- `kp::Float64 = 3.0` proportional constant for lane tracking
- `kd::Float64 = 2.0` derivative constant for lane tracking
"""
mutable struct BafflingLateralTracker <: LateralDriverModel
    a::Float64 # predicted acceleration
    σ::Float64 # optional stdev on top of the model, set to zero or NaN for deterministic behavior
    kp::Float64 # proportional constant for lane tracking
    kd::Float64 # derivative constant for lane tracking

    function BafflingLateralTracker(;
        σ::Float64 = NaN,
        kp::Float64 = 3.0,
        kd::Float64 = 2.0,
        )

        retval = new()
        retval.a = NaN
        retval.σ = σ
        retval.kp = kp
        retval.kd = kd
        retval
    end
end
get_name(::BafflingLateralTracker) = "BafflingLateralTracker"
function AutomotiveDrivingModels.track_lateral!(model::BafflingLateralTracker,
                                                laneoffset::Float64, lateral_speed::Float64;
                                                r::Float64=0.2)
    model.a = -(laneoffset+r*rand())*model.kp - lateral_speed*model.kd
    model.a = sign(model.a)*min(abs(model.a), 3.5*sin(0.3))
    model
end
function AutomotiveDrivingModels.observe!(model::BafflingLateralTracker, scene::Scene, roadway::Roadway, egoid::Int)

    ego_index = findfirst(egoid, scene)
    veh_ego = scene[ego_index]
    t = veh_ego.state.posF.t # lane offset
    dt = veh_ego.state.v * sin(veh_ego.state.posF.ϕ) # rate of change of lane offset
    model.a = -t*model.kp - dt*model.kd

    model
end
function Base.rand(model::BafflingLateralTracker)
    if isnan(model.σ) || model.σ ≤ 0.0
        model.a
    else
        rand(Normal(model.a, model.σ))
    end
end
function Distributions.pdf(model::BafflingLateralTracker, a_lat::Float64)
    if isnan(model.σ) || model.σ ≤ 0.0
        Inf
    else
        pdf(Normal(model.a, model.σ), a_lat)
    end
end
function Distributions.logpdf(model::BafflingLateralTracker, a_lat::Float64)
    if isnan(model.σ) || model.σ ≤ 0.0
        Inf
    else
        logpdf(Normal(model.a, model.σ), a_lat)
    end
end
