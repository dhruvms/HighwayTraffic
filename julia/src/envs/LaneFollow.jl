module LaneFollow

using AutomotiveDrivingModels
using AutoViz
using Parameters
using Reel

export reset, step, render, save_gif, dict_to_params
export action_space, observation_space, EnvParams

include("./structures.jl")
include("./helpers.jl")
include("../agent/agent.jl")

function make_env(params::EnvParams)
    roadway = gen_straight_roadway(params.lanes, params.length)

    ego = get_initial_egostate(params, roadway)
    ego = get_by_id(ego, EGO_ID)
    lane = get_lane(roadway, ego.state.state)

    scene, models, colours = populate_others(params)
    push!(scene, Vehicle(ego))
    colours[EGO_ID] = COLOR_CAR_EGO

    EnvState(params, roadway, scene, lane.tag, models, colours)
end

function observe(env::EnvState)
    ego = env.scene[findfirst(EGO_ID, env.scene)]

    # Ego features
    d_lon = distance_from_end(env.params, ego)

    lane = get_lane(env.roadway, ego.state.state)
    in_lane = lane.tag == env.init_lane ? 1 : 0

    ego_proj = Frenet(ego.state.state.posG, env.roadway[env.init_lane], env.roadway)
    t = ego_proj.t # displacement from lane
    ϕ = ego_proj.ϕ # angle relative to lane

    v = ego.state.state.v
    a = ego.state.a
    δ = ego.state.δ

    # TODO: normalise?
    ego_o = [d_lon, in_lane, t, ϕ, v, a, δ]
    other_o = get_neighbour_features(env)
    o = vcat(ego_o, other_o)

    return o, in_lane, d_lon
end

function Base.reset(paramdict::Dict)
    params = dict_to_params(paramdict)
    env = make_env(params)
    o, _, _ = observe(env)

    (env, o, params)
end
function is_terminal(env::EnvState)
    done = false

    ego = env.scene[findfirst(EGO_ID, env.scene)]
    dist = distance_from_end(env.params, ego)
    road_proj = proj(ego.state.state.posG, env.roadway)

    done = done || (dist <= 0.0) # no more road left
    done = done || (ego.state.state.v < 0.0) # vehicle has negative velocity
    done = done || (abs(road_proj.curveproj.t) > DEFAULT_LANE_WIDTH/2.0) # off roadway
    done = done || is_crash(env.scene)
    done
end

function reward(env::EnvState, action::Vector{Float32})
    ego = env.scene[findfirst(EGO_ID, env.scene)]
    ego_proj = Frenet(ego.state.state.posG, env.roadway[env.init_lane], env.roadway)
    dist = distance_from_end(env.params, ego)

    reward = 1.0
    # action cost
    reward -= env.params.a_cost * abs(action[1])
    reward -= env.params.δ_cost * abs(action[2])
    # desired velocity cost
    reward -= env.params.v_cost * abs(ego.state.state.v - env.params.v_des)
    # lane follow cost
    reward -= env.params.ϕ_cost * abs(ego_proj.ϕ)
    reward -= env.params.t_cost * abs(ego_proj.t)
    # distance covered reward
    reward += 1.0 - dist

    reward
end

function AutomotiveDrivingModels.tick!(env::EnvState, action::Vector{Float32},
                                        actions::Vector{Any})
    for i in 1:length(env.scene)
        veh = env.scene[findfirst(i, env.scene)]
        if veh.id == EGO_ID
            env.scene[i] = propagate(veh, action, env.roadway, env.params.dt)
        else
            state′ = propagate(veh, actions[i], env.roadway, env.params.dt)
            env.scene[i] = Entity(state′, veh.def, veh.id)
        end
    end

    env
end

function AutomotiveDrivingModels.get_actions!(
    actions::Vector{A},
    scene::EntityFrame{S, D, I},
    roadway::R,
    models::Dict{I, M}, # id → model
    ) where {S, D, I, A, R, M <: DriverModel}


    for (i, veh) in enumerate(scene)
        if veh.id == EGO_ID
            continue
        end

        model = models[veh.id]
        observe!(model, scene, roadway, veh.id)
        actions[i] = rand(model)
    end

    actions
end

function Base.step(env::EnvState, action::Vector{Float32})
    other_actions = Array{Any}(undef, length(env.scene - 1))
    get_actions!(other_actions, env.scene, env.roadway, env.models)

    tick!(env, action, other_actions) # move to next state
    r = reward(env, action)
    o, in_lane, headway = observe(env)
    terminal = is_terminal(env)

    if Bool(terminal)
        if Bool(in_lane)
            if headway <= 0.0
                r += 10.0
            end

        else
            r -= 10.0
        end
    end
    ego = env.scene[findfirst(EGO_ID, env.scene)]
    info = [ego.state.state.posF.s, ego.state.state.posF.t,
                ego.state.state.posF.ϕ, ego.state.state.v]

    (o, r, terminal, info, copy(env))
end

function AutoViz.render(env::EnvState)
    scene = Scene()
    cam = FitToContentCamera(0.01)

    ego = env.scene[findfirst(EGO_ID, env.scene)]
    push!(scene, Vehicle(ego))
    render(scene, env.roadway, cam=cam)
end

function save_gif(envs::Vector{EnvState}, filename::String="default.gif")
    framerate = Int(1.0/envs[1].params.dt)
    frames = Reel.Frames(MIME("image/png"), fps=framerate)
    for e in envs
        push!(frames, render(e))
    end
    Reel.write(filename, frames)
end

end # module
