const EGO_ID = 1
const CAR_LENGTH = 4.0

abstract type AbstractParams end
abstract type AbstractEnv end

mutable struct EnvParams <: AbstractParams
    road::Float64 # length of roadway
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
    extra_deadends::Bool # extra deadends in other lanes
    eval::Bool # evaluation
    norm_obs::Bool # normalise observations
    hri::Bool # HRI specific test case
    curriculum::Bool # (randomised) curriculum of cars and gaps during training
    gap::Float64 # gap between cars

    ego_pos::Int # location of egovehicle, between [1, cars]
    v_des::Float64 # desired velocity
    ego_dim::Int # egovehicle feature dimension
    other_dim::Int # other vehicle feature dimension
    o_dim::Int # observation space dimension
    occupancy::Bool # use occupancy grid observation

    # parameters associated with the reward function
    j_cost::Float64
    δdot_cost::Float64
    a_cost::Float64
    v_cost::Float64
    ϕ_cost::Float64
    t_cost::Float64
    deadend_cost::Float64

    mode::String # types of other vehicles (mixed/cooperative/aggressive)
    video::Bool # save video
    write_data::Bool # save data file

    ego_model::Union{Int, Nothing} # ego model for baseline
    mpc_s::Union{Int, Nothing} # MPC lookahead
    mpc_cf::Union{Float64, Nothing} # MPC collision check fraction
    mpc_cm::Union{Int, Nothing} # MPC collision check scheme

    stopgo::Bool # add stop and go behaviour
end

mutable struct EnvState <: AbstractEnv
    params::EnvParams # simulator parameters
    roadway::Roadway{Float64}
    scene::Scene
    rec::SceneRecord

    ego::Frame{Agent}
    # Vector to be reshaped into ?x4 where each row contains commanded action
    # at that (row #) timestep, the egovehicle acceleration and steering angle
    action_state::Vector{Float64}
    init_lane::LaneTag # desired lane
    steps::Int # episode ticks
    mpc::DriverModel # mpc driver model

    # logging info
    in_lane::Bool
    lane_ticks::Int
    victim_id::Union{Int, Nothing}
    merge_tick::Int
    min_dist::Float64
    lane_dist::Vector{Float64}
    car_data::Dict{Int, Dict{String, Vector{Float64}}}
    ego_data::Dict{String, Vector{Float64}}

    other_cars::Dict{Int, DriverModel}
    colours::Dict{Int, Colorant}

    prev_shaping::Union{Float64, Nothing} # used in action reward calculation

    ego_model::Union{DriverModel, Nothing} # used for baselines
end
Base.copy(e::EnvState) = EnvState(e.params, e.roadway, deepcopy(e.scene),
                                    e.rec, e.ego, copy(e.action_state),
                                    e.init_lane, e.steps, e.mpc,
                                    e.in_lane, e.lane_ticks, e.victim_id,
                                    e.merge_tick, e.min_dist, copy(e.lane_dist),
                                    copy(e.car_data), copy(e.ego_data),
                                    copy(e.other_cars), copy(e.colours),
                                    e.prev_shaping,
                                    e.ego_model)


# action limits
action_space(params::EnvParams) = ([-4.0, -0.4], [2.0, 0.4])
function observation_space(params::EnvParams)
    if params.occupancy
        fov = 2 * params.fov + 1
        return (-Inf, Inf, (5, fov, 3)) # account for egovehicle information
    else
        return (-Inf, Inf, (params.o_dim,))
    end
end
