module LaneFollow

using AutomotiveDrivingModels
using AutoViz
using Reel
using Printf

export reset, step!, save_gif
export action_space, observation_space, EnvState

include("../agent/agent.jl")
include("./structures.jl")
include("./helpers.jl")
include("../behaviours/mpc_driver.jl")

# TODO: distance_from_end does not make sense for stadium roadways

function make_env(params::EnvParams)
    if params.stadium
        roadway = gen_stadium_roadway(params.lanes, length=params.length, width=0.0, radius=10.0)
    else
        roadway = gen_straight_roadway(params.lanes, params.length)
    end

    ego, lanetag = get_initial_egostate(params, roadway)
    if params.change
        seg = lanetag.segment
        lane = lanetag.lane
        if lane > 1
            lane = 1
        else
            lane = 2
        end
        lanetag = LaneTag(seg, lane)
    end
    veh = get_by_id(ego, EGO_ID)

    scene, models, colours = populate_others(params, roadway)
    push!(scene, Vehicle(veh))
    colours[EGO_ID] = COLOR_CAR_EGO

    action_state = [0.0f0, 0.0f0, veh.state.a, veh.state.δ]

    rec = SceneRecord(params.max_ticks, params.dt)

    EnvState(params, roadway, scene, rec, ego, action_state, lanetag,
                models, colours)
end

function observe(env::EnvState)
    # ego = env.scene[findfirst(EGO_ID, env.scene)]
    ego = get_by_id(env.ego, EGO_ID)

    lane = get_lane(env.roadway, ego.state.state)
    in_lane = lane.tag.lane == env.init_lane.lane ? 1 : 0

    true_lanetag = LaneTag(lane.tag.segment, env.init_lane.lane)
    ego_proj = Frenet(ego.state.state.posG,
                        env.roadway[true_lanetag], env.roadway)
    t = ego_proj.t # displacement from lane
    ϕ = ego_proj.ϕ # angle relative to lane

    v = ego.state.state.v
    a = ego.state.a
    δ = ego.state.δ
    action = reshape(env.action_state, 4, :)'[end, 1:2]

    # TODO: normalise?
    ego_o = [in_lane, t, ϕ, v, a, δ, action[1], action[2]]
    if env.params.cars - 1 > 0
        other_o = get_neighbour_featurevecs(env)
        o = vcat(ego_o, other_o)
        return o, in_lane
    end

    return ego_o, in_lane
end

function burn_in_sim!(env::EnvState; steps::Int=20)
    other_actions = Array{Any}(undef, length(env.scene))
    for step in 1:steps
        get_actions!(other_actions, env.scene, env.roadway, env.other_cars)
        tick!(env, [0.0f0, 0.0f0], other_actions, init=true)
    end

    env
end

function Base.reset(paramdict::Dict)
    params = dict_to_params(paramdict)
    env = make_env(params)
    burn_in_sim!(env)
    check, _ = is_terminal(env)
    while check
        env = make_env(params)
        burn_in_sim!(env)
        check, _ = is_terminal(env)
    end
    env.action_state = env.action_state[end-3:end]
    update!(env.rec, env.scene)

    o, _ = observe(env)

    (env, o)
end

function is_terminal(env::EnvState; init::Bool=false)
    done = false
    final_r = 0.0

    # ego = env.scene[findfirst(EGO_ID, env.scene)]
    ego = get_by_id(env.ego, EGO_ID)
    road_proj = proj(ego.state.state.posG, env.roadway)

    done = done || (ego.state.state.v < 0.0) # vehicle has negative velocity
    final_r -= done * 5.0

    done = done || (abs(road_proj.curveproj.t) > DEFAULT_LANE_WIDTH/2.0) # off roadway
    done = done || is_crash(env, init=init)
    final_r -= done * 10.0

    if !env.params.stadium
        dist = distance_from_end(env.params, ego)
        done = done || (dist <= 0.05) # no more road left
        final_r += done * 10.0
    end

    done, final_r
end

function reward(env::EnvState, action::Vector{Float32})
    # ego = env.scene[findfirst(EGO_ID, env.scene)]
    ego = get_by_id(env.ego, EGO_ID)
    lane = get_lane(env.roadway, ego.state.state)
    true_lanetag = LaneTag(lane.tag.segment, env.init_lane.lane)

    ego_proj = Frenet(ego.state.state.posG,
                        env.roadway[true_lanetag], env.roadway)

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

    if !env.params.stadium
        dist = distance_from_end(env.params, ego)
        reward += 1.0 - dist
    end

    reward
end

function AutomotiveDrivingModels.tick!(env::EnvState, action::Vector{Float32},
                                        actions::Vector{Any}; init::Bool=false)
    ego = get_by_id(env.ego, EGO_ID)
    a = ego.state.a
    δ = ego.state.δ

    done = false
    for (i, veh) in enumerate(env.scene)
        if veh.id == EGO_ID
            if !init
                state′, done = propagate(ego, action, env.roadway, env.params.dt)
                env.scene[findfirst(EGO_ID, env.scene)] = Vehicle(state′)
                env.ego = Frame([state′])

                ego = get_by_id(env.ego, EGO_ID)
                a = ego.state.a
                δ = ego.state.δ
            end
        else
            state′ = propagate(veh, actions[i], env.roadway, env.params.dt)
            env.scene[findfirst(veh.id, env.scene)] = Entity(state′, veh.def, veh.id)
        end
    end

    env.action_state = vcat(env.action_state, append!(action, [a, δ]))
    (env, done)
end

function AutomotiveDrivingModels.get_actions!(
    actions::Vector{A},
    scene::EntityFrame{S, D, I},
    roadway::R,
    models::Dict{I, M}, # id → model
    ) where {S, D, I, A, R, M <: DriverModel}


    for (i, veh) in enumerate(scene)
        if veh.id == EGO_ID
            actions[i] = LatLonAccel(0, 0)
            continue
        end

        model = models[veh.id]
        AutomotiveDrivingModels.observe!(model, scene, roadway, veh.id)
        actions[i] = rand(model)
    end

    actions
end

function step!(env::EnvState, action::Vector{Float32})
    other_actions = Array{Any}(undef, length(env.scene))
    get_actions!(other_actions, env.scene, env.roadway, env.other_cars)

    env, done = tick!(env, action, other_actions) # move to next state
    update!(env.rec, env.scene)
    r = reward(env, action)
    o, in_lane = observe(env)
    terminal, final_r = is_terminal(env)
    terminal = terminal || done

    if Bool(terminal)
        r += final_r
        if Bool(in_lane)
            r += 1.0
        end
    end

    ego = env.scene[findfirst(EGO_ID, env.scene)]
    info = [ego.state.posF.s, ego.state.posF.t,
                ego.state.posF.ϕ, ego.state.v]

    (o, r, terminal, info, copy(env))
end

function save_gif(env::EnvState, filename::String="default.gif")
    framerate = Int(1.0/env.params.dt)
    frames = Reel.Frames(MIME("image/png"), fps=framerate)

    cam = CarFollowCamera(EGO_ID, 20.0)
    ego = get_by_id(env.ego, EGO_ID)

    ticks = nframes(env.rec)
    for frame_index in 1:ticks
        scene = env.rec[frame_index-ticks]
        ego = scene[findfirst(EGO_ID, scene)]

        action_state = reshape(env.action_state, 4, :)'
        action_state = action_state[frame_index, :]
        jerk_text = @sprintf("Jerk:  %2.2f m/s^3", action_state[1])
        δrate_text = @sprintf("δ rate:  %2.2f rad/s", action_state[2])
        acc_text = @sprintf("acc:  %2.2f m/s^2", action_state[3])
        δ_text = @sprintf("δ:  %2.2f rad", action_state[4])
        v_text = @sprintf("v:  %2.2f m/s", ego.state.v)
        action_overlay = TextOverlay(text=[jerk_text, δrate_text,
                            acc_text, δ_text, v_text], font_size=18)

        push!(frames, render(scene, env.roadway, [action_overlay], cam=cam, car_colors=env.colours))
    end
    Reel.write(filename, frames)
end

end # module
