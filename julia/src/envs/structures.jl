const EGO_ID = 1

mutable struct EnvParams
    length::Float64 # length of roadway
    lanes::Int # number of lanes in roadway
    cars::Int # number of cars on roadway, including egovehicle
    v_des::Float64 # desired velocity
    dt::Float64 # timestep
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

mutable struct EnvState
    params::EnvParams
    roadway::Roadway{Float64}
    scene::Scene

    ego::Frame{Agent}
    action::Vector{Float32}
    init_lane::LaneTag

    other_cars::Dict{Int, DriverModel}
    colours::Dict{Int, Colorant}
end
Base.copy(e::EnvState) = EnvState(e.params, e.roadway, deepcopy(e.scene), e.ego,
                                    e.action, e.init_lane, e.other_cars, e.colours)

action_space(params::EnvParams) = ([-4.0, -0.4], [2.0, 0.4])
observation_space(params::EnvParams) = (fill(-Inf, params.o_dim), fill(Inf, params.o_dim))
