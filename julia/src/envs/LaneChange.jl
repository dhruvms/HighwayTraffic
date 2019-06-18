module LaneChange

using AutomotiveDrivingModels
using AutoViz
using Parameters

export reset, step, render, dict_to_params
export action_space, observation_space, EnvParams

const EGO_ID = 1

mutable struct EnvParams
    length::Float64 # length of roadway
    lanes::Int64 # number of lanes in roadway
    cars::Int64 # number of cars on roadway, including egovehicle
    v_des::Float64 # desired velocity
    dt::Float64 # timestep
    o_dim::Int64 # observation space dimension

    # TODO: add any reward params here
    a_cost::Float64
    δ_cost::Float64
    v_cost::Float64
    ϕ_cost::Float64
    t_cost::Float64
end

include("../agent/agent.jl")
include("./helpers.jl")

mutable struct EnvState
    params::EnvParams
    roadway::Roadway{Float64}

    agent::Frame{Agent}
    init_lane::LaneTag
    des_lane::LaneTag

    # TODO: add something for other cars here
end

action_space(params::EnvParams) = ([-4.0, -0.4], [2.0, 0.4])
observation_space(params::EnvParams) = (fill(-Inf, params.o_dim), fill(Inf, params.o_dim))

function dict_to_params(params::Dict)
    length = get(params, "length", 200.0)
    lanes = get(params, "lanes", 2)
    cars = get(params, "cars", 1)
    v_des = get(params, "v_des", 10.0)
    dt = get(params, "dt", 0.2)
    o_dim = get(params, "o_dim", 8)

    a_cost = get(params, "a_cost", 0.1)
    δ_cost = get(params, "d_cost", 0.01)
    v_cost = get(params, "v_cost", 0.2)
    ϕ_cost = get(params, "phi_cost", 0.35)
    t_cost = get(params, "t_cost", 0.34)

    EnvParams(length, lanes, cars, v_des, dt, o_dim,
                a_cost, δ_cost, v_cost, ϕ_cost, t_cost)
end

function get_initial_egostate(params::EnvParams, roadway::Roadway{Float64})
    v0 = rand() * params.v_des
    s0 = rand() * (params.length / 4.0)
    ego = Entity(AgentState(roadway, v=v0, s=s0), EgoVehicle(), EGO_ID)
    return Frame([ego])
end

function make_env(params::EnvParams)
    roadway = gen_straight_roadway(params.lanes, params.length)
    ego = get_initial_egostate(params, roadway)
    veh = get_by_id(ego, EGO_ID)
    lane = get_lane(roadway, veh.state.state)

    # TODO: get desired lane
    left_exists, left_lane = try
        true, roadway[LaneTag(lane.tag.segment, lane.tag.lane + 1)]
    catch
        false, nothing
    end
    right_exists, right_lane = try
        true, roadway[LaneTag(lane.tag.segment, lane.tag.lane - 1)]
    catch
        false, nothing
    end

    des_lane = nothing
    if left_exists && !right_exists
        des_lane = left_lane
    elseif right_exists && !left_exists
        des_lane = right_lane
    elseif left_exists && right_exists
        if rand() >= 0.5
            des_lane = left_lane
        else
            des_lane = right_lane
        end
    else
        des_lane = lane
    end


    EnvState(params, roadway, ego, lane.tag, des_lane.tag)
end

function observe(env::EnvState)
    veh = get_by_id(env.agent, EGO_ID)

    # Ego features
    d_lon = distance_from_end(env.params, veh)

    lane = get_lane(env.roadway, veh.state.state)
    in_init_lane = lane.tag == env.init_lane ? 1 : 0
    in_des_lane = lane.tag == env.des_lane ? 1 : 0

    veh_proj = Frenet(veh.state.state.posG, env.roadway[env.des_lane], env.roadway)
    t = veh_proj.t # displacement from lane
    ϕ = veh_proj.ϕ # angle relative to lane

    v = veh.state.state.v
    a = veh.state.a
    δ = veh.state.δ

    # TODO: normalise?
    o = [d_lon, in_init_lane, in_des_lane, t, ϕ, v, a, δ]
    return o, in_init_lane, in_des_lane, d_lon
end

function Base.reset(paramdict::Dict)
    params = dict_to_params(paramdict)
    env = make_env(params)
    o, _, _, _ = observe(env)

    (env, o, params)
end

function tick!(env::EnvState, action::Vector{Float32})
    veh = get_by_id(env.agent, EGO_ID)
    env.agent = Frame([propagate(veh, action, env.roadway, env.params.dt)])
    env
end

function reward(env::EnvState, action::Vector{Float32})
    veh = get_by_id(env.agent, EGO_ID)
    veh_proj = Frenet(veh.state.state.posG, env.roadway[env.des_lane], env.roadway)
    dist = distance_from_end(env.params, veh)

    reward = 1.0
    # action cost
    reward -= env.params.a_cost * abs(action[1])
    reward -= env.params.δ_cost * abs(action[2])
    # desired velocity cost
    reward -= env.params.v_cost * abs(veh.state.state.v - env.params.v_des)
    # lane change cost - increases with decreasing distance
    reward -= (env.params.ϕ_cost * abs(veh_proj.ϕ)) / (max(dist, 0.001))
    reward -= (env.params.t_cost * abs(veh_proj.t)) / (max(dist, 0.001))

    reward
end

function is_terminal(env::EnvState)
    done = false

    veh = get_by_id(env.agent, EGO_ID)
    dist = distance_from_end(env.params, veh)
    road_proj = proj(veh.state.state.posG, env.roadway)

    done = done || (dist <= 0.0) # no more road left
    done = done || (veh.state.state.v < 0.0) # vehicle has negative velocity
    done = done || (abs(road_proj.curveproj.t) > DEFAULT_LANE_WIDTH/2.0) # off roadway
    done
end

function Base.step(env::EnvState, action::Vector{Float32})
    tick!(env, action) # move to next state
    r = reward(env, action)
    o, in_init_lane, in_des_lane, headway = observe(env)
    terminal = is_terminal(env)

    if Bool(terminal)
        if Bool(in_init_lane)
            if headway <= 0.0
                r += 2.0
            end
        end
        if Bool(in_des_lane)
            if headway <= 0.0
                r += 10.0
            end
        end
    end
    veh = get_by_id(env.agent, EGO_ID)
    info = [veh.state.state.posF.s, veh.state.state.posF.t,
                veh.state.state.posF.ϕ, veh.state.state.v,
                in_init_lane, in_des_lane]

    (o, r, terminal, info, env)
end

function AutoViz.render(env::EnvState)
    scene = Scene()
    cam = FitToContentCamera(0.01)

    veh = get_by_id(env.agent, EGO_ID)
    push!(scene, veh)
    render!(scene, env.roadway, cam=cam)
end

end # module
