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
    room::Float64 # room between cars
    stadium::Bool # stadium roadway
    change::Bool # change to different lane

    ego_pos::Int # location of egovehicle, between [1, cars]
    v_des::Float64 # desired velocity
    ego_dim::Int # egovehicle feature dimension
    other_dim::Int # other vehicle feature dimension
    o_dim::Int # observation space dimension

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
    action_state::Vector{Float32}
    init_lane::LaneTag

    other_cars::Dict{Int, DriverModel}
    colours::Dict{Int, Colorant}

    # EnvState() = new()
end
Base.copy(e::EnvState) = EnvState(e.params, e.roadway, deepcopy(e.scene),
                                    e.rec, e.ego, copy(e.action_state),
                                    e.init_lane, e.other_cars, e.colours)


action_space(params::EnvParams) = ([-4.0, -0.4], [2.0, 0.4])
observation_space(params::EnvParams) = (fill(-Inf, params.o_dim),
                                                        fill(Inf, params.o_dim))

mutable struct TCNParams <: AbstractParams
    length::Float64 # length of roadway
    lanes::Int # number of lanes in roadway
    cars::Int # number of cars on roadway, including egovehicle
    dt::Float64 # timestep
    max_ticks::Int # max ticks per episode
    room::Float64 # room between cars
    stadium::Bool # stadium roadway

    v_des::Float64 # max desired velocity
    sampled::Int # number of cars for data sampling
    features::Int # feature dimension
    max_neighbours::Int # max neighbours to consider
    radius::Float64 # neighbour distance radius
end

mutable struct TCNEnv <: AbstractEnv
    params::TCNParams
    roadway::Roadway{Float64}
    scene::Scene
    rec::SceneRecord

    car_models::Dict{Int, DriverModel}
    car_colours::Dict{Int, Colorant}

    car_ids::Vector{Int}

    # car id => ordered vector of neighbour ids
    # (ordering is same as row ordering in feature matrix)
    prev_neighbours::Dict{Int, Vector{Int}}
    # car id => 2D matrix of neighbour features (one row per neighbour)
    prev_features::Dict{Int, Array{Float64, 2}}

    DATA::Array{Float64, 2}
end
Base.copy(e::TCNEnv) = TCNEnv(e.params, e.roadway, deepcopy(e.scene), e.rec,
                                e.car_models, e.car_colours,
                                copy(e.car_ids),
                                copy(e.prev_neighbours), copy(e.prev_features),
                                copy(e.DATA))
