module LaneFollow

using AutomotiveDrivingModels
using AutoViz
using Reel
using Printf

export reset, step, render, save_gif, dict_to_params
export action_space, observation_space, EnvParams

include("../agent/agent.jl")
include("./structures.jl")
include("./helpers.jl")
include("../behaviours/mpc_driver.jl")

# TODO: distance_from_end does not make sense for stadium roadways

function make_env(params::EnvParams)
    roadway = nothing
    if params.stadium
        roadway = gen_stadium_roadway(params.lanes, length=params.length)
    else
        roadway = gen_straight_roadway(params.lanes, params.length)
    end

    ego, lanetag = get_initial_egostate(params, roadway)
    veh = get_by_id(ego, EGO_ID)

    scene, models, colours = populate_others(params, roadway)
    push!(scene, Vehicle(veh))
    colours[EGO_ID] = COLOR_CAR_EGO

    action = [0.0f0, 0.0f0]

    EnvState(params, roadway, scene, ego, action, lanetag, models, colours)
end

function observe(env::EnvState)
    # ego = env.scene[findfirst(EGO_ID, env.scene)]
    ego = get_by_id(env.ego, EGO_ID)

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
    ego_o = [d_lon, in_lane, t, ϕ, v, a, δ, env.action[1], env.action[2]]
    if env.params.cars - 1 > 0
        other_o = get_neighbour_features(env)
        o = vcat(ego_o, other_o)
        return o, in_lane, d_lon
    end

    return ego_o, in_lane, d_lon
end

function Base.reset(paramdict::Dict)
    params = dict_to_params(paramdict)
    env = make_env(params)
    while is_terminal(env, init=true)
        env = make_env(params)
    end
    o, _, _ = observe(env)

    (env, o, params)
end
function is_terminal(env::EnvState; init::Bool=false)
    done = false

    # ego = env.scene[findfirst(EGO_ID, env.scene)]
    ego = get_by_id(env.ego, EGO_ID)
    dist = distance_from_end(env.params, ego)
    road_proj = proj(ego.state.state.posG, env.roadway)

    done = done || (dist <= 0.0) # no more road left
    done = done || (ego.state.state.v < 0.0) # vehicle has negative velocity
    done = done || (abs(road_proj.curveproj.t) > DEFAULT_LANE_WIDTH/2.0) # off roadway
    done = done || is_crash(env, init=init)

    done
end

function reward(env::EnvState, action::Vector{Float32})
    # ego = env.scene[findfirst(EGO_ID, env.scene)]
    ego = get_by_id(env.ego, EGO_ID)
    ego_proj = Frenet(ego.state.state.posG, env.roadway[env.init_lane], env.roadway)
    dist = distance_from_end(env.params, ego)

    reward = 1.0
    # action cost
    reward -= env.params.j_cost * abs(action[1])
    reward -= env.params.δdot_cost * abs(action[2])
    reward -= env.params.a_cost * abs(ego.state.a)
    # desired velocity cost
    reward -= env.params.v_cost * abs(ego.state.state.v - env.params.v_des)
    # lane follow cost
    reward -= env.params.ϕ_cost * abs(ego_proj.ϕ)
    reward -= env.params.t_cost * abs(ego_proj.t)
    # distance covered reward
    reward += 1.0 - dist

    # if env.params.cars - 1 > 0
    #     other_o = get_neighbour_features(env)
    #     reward += sum(abs.(other_o[(abs.(other_o) .!= 0.0) .& (abs.(other_o) .!= 1.0)]))
    # end

    reward
end

function AutomotiveDrivingModels.tick!(env::EnvState, action::Vector{Float32},
                                        actions::Vector{Any})
    for i in 1:length(env.scene)
        veh = env.scene[i]
        if veh.id == EGO_ID
            ego = get_by_id(env.ego, EGO_ID)
            state′ = propagate(ego, action, env.roadway, env.params.dt)
            env.scene[i] = Vehicle(state′)
            env.ego = Frame([state′])
        else
            state′ = propagate(veh, actions[i], env.roadway, env.params.dt)
            env.scene[i] = Entity(state′, veh.def, veh.id)
        end
    end

    env.action = action
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
        AutomotiveDrivingModels.observe!(model, scene, roadway, veh.id)
        actions[i] = rand(model)
    end

    actions
end

function Base.step(env::EnvState, action::Vector{Float32})
    other_actions = Array{Any}(undef, length(env.scene) - 1)
    get_actions!(other_actions, env.scene, env.roadway, env.other_cars)

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
    info = [ego.state.posF.s, ego.state.posF.t,
                ego.state.posF.ϕ, ego.state.v]

    (o, r, terminal, info, copy(env))
end

function AutoViz.render(env::EnvState)
    cam = FitToContentCamera(0.01)

    jerk_text = @sprintf("Jerk:  %2.2f m/s^3", env.action[1])
    δrate_text = @sprintf("δ rate:  %2.2f rad/s", env.action[2])
    action_overlay = TextOverlay(text=[jerk_text, δrate_text], font_size=18)
    render(env.scene, env.roadway, [action_overlay], cam=cam, car_colors=env.colours)
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
