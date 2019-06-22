using Parameters

"""
    EgoVehicle <: AbstractAgentDefinition
Object representing the physical characteristics and actuation capabilities
of an Autonomous Vehicle.
# Constructor
`EgoVehicle(kwargs...)`
# Fields
    - `class::Int64 = AgentClass.CAR`
    - `a_max::Float64 = 3.5` [m/s^2]
    - `a_min::Float64 = -4.0` [m/s^2]
    - `j_max::Float64 = 2.0` [m/s^3]
    - `j_min::Float64 = -4.0` [m/s^3]
    ## steering angle
    - `δ_max = 0.6`  [rad] (34.3 deg)
    - `d_dot_max = 0.4` [rad/s] (22.9deg/s)
    ## bicyle model params
    - `l_f::Float64 = 1.5`  distance between cg and front axle [m]
    - `l_r::Float64 = 1.5`  distance between cg and rear axle [m]

    ## box params
    - `length::Float64 = 4.0` [m]
    - `width::Float64 = 1.8` [m]
"""
@with_kw struct EgoVehicle <: AbstractAgentDefinition
    class::Int64 = AgentClass.CAR

    # acceleration
    a_max::Float64 = 3.5 # m/s^2
    a_min::Float64 = -4.0 # m/s^2
    j_max::Float64 = 2.0 # m/s^3
    j_min::Float64 = -4.0 # m/s^3

    # steering angle
    δ_max = 0.6 # [rad] (34.3 deg)
    δ_dot_max = 0.4 # [rad/s] (22.9deg/s)

    # bicyle model params
    l_f::Float64 = 1.5 # distance between cg and front axle [m]
    l_r::Float64 = 1.5 # distance between cg and rear axle [m]

    # box params
    length::Float64 = 4.0
    width::Float64 = 1.8
end

AutomotiveDrivingModels.VehicleDef(def::EgoVehicle) = VehicleDef(def.class, def.length, def.width)

Base.length(veh::EgoVehicle) = veh.length
AutomotiveDrivingModels.width(veh::EgoVehicle) = veh.width

"""
    AgentState
structure representing the state of an agent for lane changing problems.
# Fields
    - `state::VehicleState` see AutomotiveDrivingModels.jl
    - `acc::Float64` acceleration [m/s^2]
    - `δ::Float64` steering angle [rad]
"""
struct AgentState
    state::VehicleState
    a::Float64 # acceleration [m/s^2]
    δ::Float64 # steering angle [rad]
end

function AgentState(roadway::Roadway;
                    s::Float64 = 0.0,
                    t::Float64 = 0.0,
                    ϕ::Float64 = 0.0,
                    v::Float64 = 0.0,
                    a::Float64 = 0.0,
                    δ::Float64 = 0.0,
                    lane::LaneTag = LaneTag(1,1))
    posF = Frenet(roadway[lane], s, t, ϕ)
    state = VehicleState(posF, roadway, v)
    return AgentState(state, a, δ)
end

function AutomotiveDrivingModels.Vehicle(veh::Entity{AgentState,EgoVehicle,Int64})
    return Vehicle(veh.state.state, VehicleDef(veh.def), veh.id)
end

function AutoViz.render!(rendermodel::RenderModel,
                         veh::Entity{AgentState, EgoVehicle, Int64},
                         color::Colorant)
    return render!(rendermodel, Vehicle(veh), color)
end

const Agent = Entity{AgentState, EgoVehicle, Int64}

function AutomotiveDrivingModels.propagate(veh::Entity{AgentState,EgoVehicle,Int64},
                           action::Vector{Float32}, roadway::Roadway, dt::Float64)
    # update acceleration
    a_ = veh.state.a + action[1] * dt
    a_ = clamp(a_, veh.def.a_min, veh.def.a_max)

    # update angle
    δ_ = veh.state.δ + action[2] * dt
    δ_ = clamp(δ_, -veh.def.δ_max, veh.def.δ_max)

    # run bicycle model
    L = veh.def.l_f + veh.def.l_r
    l = -veh.def.l_r

    a = a_ # accel [m/s²]
    δ = δ_ # steering wheel angle [rad]

    agt = veh.state
    x = agt.state.posG.x
    y = agt.state.posG.y
    θ = agt.state.posG.θ
    v = agt.state.v

    s = v*dt + a*dt*dt/2 # distance covered
    s = max(0, s) # no back up
    v_ = v + a*dt
    v_ = max(0, v_) # no back up

    if δ ≈ 0.0 # just drive straight
        posG = agt.state.posG + polar(s, θ)
    else # drive in circle
        R = L/tan(δ) # turn radius

        β = s/R
        xc = x - R*sin(θ) + l*cos(θ)
        yc = y + R*cos(θ) + l*sin(θ)

        θ_ = mod(θ+β, 2π)
        x_ = xc + R*sin(θ+β) - l*cos(θ_)
        y_ = yc - R*cos(θ+β) - l*sin(θ_)

        posG = VecSE2(x_, y_, θ_)
    end

    return Entity(AgentState(VehicleState(posG, roadway, v_), a, δ), veh.def, veh.id)
end
