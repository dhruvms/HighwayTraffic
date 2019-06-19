const EGO_ID = 1

mutable struct EnvParams
    length::Float64 # length of roadway
    lanes::Int # number of lanes in roadway
    cars::Int # number of cars on roadway, including egovehicle
    v_des::Float64 # desired velocity
    dt::Float64 # timestep
    o_dim::Int # observation space dimension

    a_cost::Float64
    δ_cost::Float64
    v_cost::Float64
    ϕ_cost::Float64
    t_cost::Float64

    # other driver parameters
    num_others::Int
    random::Bool
end

mutable struct EnvState
    params::EnvParams
    roadway::Roadway{Float64}
    scene::Scene

    ego::Frame{Agent}
    init_lane::LaneTag

    # TODO: add something for other cars here
    other_cars::Dict{Int, DriverModel}
    colours::Dict{Int, Colorant}
end
Base.copy(e::EnvState) = EnvState(e.params, e.roadway, deepcopy(e.scene), e.ego,
                                    e.init_lane, e.other_cars, e.colours)

action_space(params::EnvParams) = ([-4.0, -0.4], [2.0, 0.4])
observation_space(params::EnvParams) = (fill(-Inf, params.o_dim), fill(Inf, params.o_dim))
