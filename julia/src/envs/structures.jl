const EGO_ID = 1
const CAR_LENGTH = 4.0

abstract type AbstractParams end
abstract type AbstractEnv end

mutable struct EnvParams <: AbstractParams
    length::Float64 # length of roadway
    lanes::Int # number of lanes in roadway
    cars::Int # number of cars on roadway, including egovehicle
    dt::Float64 # timestep
    max_ticks::Int # max ticks per episode
    rooms::Array{Float64, 2} # room between cars
    stadium::Bool # stadium roadway
    change::Bool # change to different lane
    both::Bool # change or follow
    fov::Int # longitudinal field-of-view
    beta::Bool # beta distribution policy in use
    clamp::Bool # clamp action to limits

    ego_pos::Int # location of egovehicle, between [1, cars]
    v_des::Float64 # desired velocity
    ego_dim::Int # egovehicle feature dimension
    other_dim::Int # other vehicle feature dimension
    o_dim::Int # observation space dimension
    occupancy::Bool # use occupancy grid observation

    j_cost::Float64
    δdot_cost::Float64
    a_cost::Float64
    v_cost::Float64
    ϕ_cost::Float64
    t_cost::Float64
end

mutable struct EnvState <: AbstractEnv
    params::EnvParams
    roadway::Roadway{Float64}
    scene::Scene
    rec::SceneRecord

    ego::Frame{Agent}
    # Vector to be reshaped into ?x4 where each row contains commanded action
    # at that (row #) timestep, the egovehicle acceleration and steering angle
    action_state::Vector{Float64}
    init_lane::LaneTag
    steps::Int

    other_cars::Dict{Int, DriverModel}
    colours::Dict{Int, Colorant}

    # EnvState() = new()
end
Base.copy(e::EnvState) = EnvState(e.params, e.roadway, deepcopy(e.scene),
                                    e.rec, e.ego, copy(e.action_state),
                                    e.init_lane, e.steps,
                                    e.other_cars, e.colours)


action_space(params::EnvParams) = ([-4.0, -0.4], [2.0, 0.4])
function observation_space(params::EnvParams)
    if params.occupancy
        fov = 2 * params.fov + 1
        return (-Inf, Inf, (5, fov, 3))
    else
        return (-Inf, Inf, (params.o_dim,))
    end
end
