module LaneFollow

using AutomotiveDrivingModels
using AutoViz
using Reel
using Printf
using DelimitedFiles

export reset, step!, save_gif
export action_space, observation_space, EnvState

include("../agent/agent.jl")
include("./structures.jl")
include("./helpers.jl")
include("../behaviours/mpc_driver.jl")
include("../behaviours/baffling_drivers.jl")
include("./traj_overlay.jl")

# TODO: distance_from_end does not make sense for stadium roadways

function make_env(params::EnvParams)
    if params.stadium
        roadway = gen_stadium_roadway(params.lanes, length=params.length, width=0.0, radius=10.0)
    else
        roadway = gen_straight_roadway(params.lanes, params.length)
    end

    ego, lanetag = get_initial_egostate(params, roadway)
    seg = lanetag.segment
    lane = lanetag.lane
    if params.change
        lane = try
            rand(filter(l->l != lanetag.lane, 1:params.lanes))
        catch
            1
        end
    elseif params.both
        lane = rand(1:params.lanes)
    end
    lanetag = LaneTag(seg, lane)
    veh = get_by_id(ego, EGO_ID)
    ego_proj = Frenet(veh.state.state.posG, roadway[lanetag], roadway)

    action_state = [0.0, 0.0, veh.state.a, veh.state.δ]
    ego_data = Dict{String, Vector{Float64}}(
                    "deviation" => [abs(ego_proj.t)],
                    "jerk" => [action_state[1]],
                    "steer_rate" => [action_state[2]],
                    "acc" => [action_state[3]],
                    "vel" => [veh.state.state.v],
                    "steer_angle" => [action_state[4]])
    scene, models, colours = populate_scene(params, roadway, veh)
    rec = SceneRecord(params.max_ticks, params.dt)
    steps = 0
    mpc = MPCDriver(params.dt)

    in_lane = false
    lane_ticks = 0
    victim_id = nothing
    merge_tick = 0
    car_data = Dict{Int, Dict{String, Vector{Float64}}}()
    for (i, veh) in enumerate(scene)
        if veh.id ≠ EGO_ID && veh.id <= 100
            car_data[veh.id] = Dict{String, Vector{Float64}}(
                                    "vel" => [veh.state.v],
                                    "alat" => [0.0],
                                    "alon" => [0.0])
        end
    end

    EnvState(params, roadway, scene, rec, ego, action_state, lanetag, steps,
                mpc,
                in_lane, lane_ticks, victim_id,
                merge_tick, car_data, ego_data,
                models, colours)
end

function observe(env::EnvState)
    ego_o = get_ego_features(env)
    if env.params.cars - 1 > 0
        other_o = get_neighbour_featurevecs(env)
        o = vcat(ego_o, other_o)
        return o, ego_o[1], ego_o[2]
    end

    return ego_o, ego_o[1], ego_o[2]
end

function observe_occupancy(env::EnvState)
    ego_o = get_ego_features(env)
    o = get_occupancy_image(env)

    fov = 2 * env.params.fov + 1
    ego_mat = zeros(fov, 3)
    ego_mat[1:env.params.ego_dim, 1] = ego_o
    o = cat(o, ego_mat, dims=3)

    return o, ego_o[1], ego_o[2]
end

function burn_in_sim!(env::EnvState; steps::Int=0)
    other_actions = Array{Any}(undef, length(env.scene))
    for step in 1:steps
        get_actions!(other_actions, env.scene, env.roadway, env.other_cars)
        tick!(env, [0.0f0, 0.0f0], other_actions, init=true)
    end

    if steps > 0 && env.params.eval
        for (i, veh) in enumerate(env.scene)
            if veh.id ≠ EGO_ID && veh.id ∈ keys(env.car_data)
                env.car_data[veh.id] = Dict{String, Vector{Float64}}(
                                        "vel" => [veh.state.v],
                                        "alat" => [other_actions[i].a_lat],
                                        "alon" => [other_actions[i].a_lon])
            end
        end
    end

    env
end

function Base.reset(paramdict::Dict)
    params = dict_to_simparams(paramdict)
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

    if env.params.occupancy
        o, _, _ = observe_occupancy(env)
    else
        o, _, _ = observe(env)
    end

    (env, o)
end

function is_terminal(env::EnvState; init::Bool=false)
    done = false
    final_r = 0.0

    # ego = env.scene[findfirst(EGO_ID, env.scene)]
    ego = get_by_id(env.ego, EGO_ID)
    road_proj = proj(ego.state.state.posG, env.roadway)

    # done = done || (ego.state.state.v < 0.0) # vehicle has negative velocity
    done = done || (abs(road_proj.curveproj.t) > DEFAULT_LANE_WIDTH/2.0) # off roadway
    done = done || is_crash(env, init=init)
    final_r -= done * 100.0
    done = done || (env.steps ≥ env.params.max_ticks)

    # max_s = 0.0
    # for (i, veh) in enumerate(env.scene)
    #     if veh.id <= 100 && veh.state.posF.s > max_s
    #         max_s = veh.state.posF.s
    #     end
    # end
    # done = done || (max_s ≥ env.params.length * 0.95)

    done, final_r
end

function reward(env::EnvState, action::Vector{Float64},
                    deadend::Float64, in_lane::Bool)
    # ego = env.scene[findfirst(EGO_ID, env.scene)]
    ego = get_by_id(env.ego, EGO_ID)
    lane = get_lane(env.roadway, ego.state.state)

    true_lanetag = LaneTag(lane.tag.segment, env.init_lane.lane)
    ego_proj = Frenet(ego.state.state.posG,
                        env.roadway[true_lanetag], env.roadway)

    reward = 0.0
    # # action cost
    # reward -= env.params.j_cost * abs(action[1])
    # reward -= env.params.δdot_cost * abs(action[2])
    # reward -= env.params.a_cost * abs(ego.state.a)
    # desired velocity cost
    reward -= env.params.v_cost * abs(ego.state.state.v - env.params.v_des)
    # lane follow cost
    reward -= env.params.t_cost * abs(ego_proj.t)
    reward -= env.params.ϕ_cost * abs(ego_proj.ϕ) * in_lane
    # distance to deadend
    if in_lane
        reward += 1.0
        reward += env.params.deadend_cost * deadend

        if env.params.eval
            if env.in_lane
                env.lane_ticks += 1
            else
                env.in_lane = true
                env.lane_ticks = 1
            end

            if env.lane_ticks ≥ 10 && isnothing(env.victim_id)
                env.merge_tick = env.steps
                victim = get_neighbor_rear_along_lane(
                            env.scene, EGO_ID, env.roadway,
                            VehicleTargetPointFront(), VehicleTargetPointFront(),
                            VehicleTargetPointRear())
                if !isnothing(victim.ind)
                    env.victim_id = victim.ind

                    for car in keys(env.car_data)
                        if car ≠ env.victim_id
                            delete!(env.car_data, car)
                        end
                    end
                end
            end
        end
    else
        reward -= env.params.deadend_cost * (1.0 - deadend)

        if env.params.eval
            if env.in_lane
                env.in_lane = false
                env.lane_ticks = 0
            end
        end
    end

    reward
end

function AutomotiveDrivingModels.tick!(env::EnvState, action::Vector{Float64},
                                        actions::Vector{Any}; init::Bool=false)
    ego = get_by_id(env.ego, EGO_ID)
    a = ego.state.a
    δ = ego.state.δ

    neg_v = false
    for (i, veh) in enumerate(env.scene)
        if veh.id == EGO_ID
            if !init
                ego′, neg_v = propagate(ego, action, env.roadway, env.params.dt)
                env.scene[findfirst(EGO_ID, env.scene)] = Vehicle(ego′)
                env.ego = Frame([ego′])

                ego = get_by_id(env.ego, EGO_ID)
                a = ego.state.a
                δ = ego.state.δ

                if env.params.eval
                    lane = get_lane(env.roadway, ego.state.state)

                    true_lanetag = LaneTag(lane.tag.segment, env.init_lane.lane)
                    ego_proj = Frenet(ego.state.state.posG,
                                        env.roadway[true_lanetag], env.roadway)

                    push!(env.ego_data["deviation"], abs(ego_proj.t))
                    push!(env.ego_data["jerk"], action[1])
                    push!(env.ego_data["steer_rate"], action[2])
                    push!(env.ego_data["acc"], a)
                    push!(env.ego_data["vel"], ego′.state.state.v)
                    push!(env.ego_data["steer_angle"], δ)
                end
            end
        elseif veh.id <= 100
            state′ = propagate(veh, actions[i], env.roadway, env.params.dt)
            env.scene[findfirst(veh.id, env.scene)] = Entity(state′, veh.def, veh.id)

            if env.params.eval
                if veh.id ≠ EGO_ID && veh.id ∈ keys(env.car_data)
                    push!(env.car_data[veh.id]["vel"], state′.v)
                    push!(env.car_data[veh.id]["alat"], actions[i].a_lat)
                    push!(env.car_data[veh.id]["alon"], actions[i].a_lon)
                end
            end
        end
    end

    env.action_state = vcat(env.action_state, append!(action, [a, δ]))
    (env, neg_v)
end

function AutomotiveDrivingModels.get_actions!(
    actions::Vector{A},
    scene::EntityFrame{S, D, I},
    roadway::R,
    models::Dict{I, M}, # id → model
    ) where {S, D, I, A, R, M <: DriverModel}


    for (i, veh) in enumerate(scene)
        if veh.id == EGO_ID || veh.id >= 101
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
    action = convert(Vector{Float64}, action)
    action_lims = action_space(env.params)
    if env.params.beta
        action[1] = map_to_range(action[1], 0.0, 1.0,
                                    action_lims[1][1], action_lims[2][1])
        action[2] = map_to_range(action[2], 0.0, 1.0,
                                    action_lims[1][2], action_lims[2][2])
    end
    if env.params.clamp
        action = [clamp(action[1], action_lims[1][1], action_lims[2][1]),
                    clamp(action[2], action_lims[1][2], action_lims[2][2])]
    end

    other_actions = Array{Any}(undef, length(env.scene))
    get_actions!(other_actions, env.scene, env.roadway, env.other_cars)

    env, neg_v = tick!(env, action, other_actions) # move to next state
    update!(env.rec, env.scene)
    env.steps += 1

    if env.params.occupancy
        o, deadend, in_lane = observe_occupancy(env)
    else
        o, deadend, in_lane = observe(env)
    end
    terminal, final_r = is_terminal(env)
    if env.params.eval
        terminal = false
    end

    r = reward(env, action, deadend, Bool(in_lane))
    r -= 2.0 * neg_v
    if Bool(terminal)
        r += final_r
    end

    ego = get_by_id(env.ego, EGO_ID)
    info = [ego.state.state.posF.s, ego.state.state.posF.t,
                ego.state.state.posF.ϕ, ego.state.state.v]

    (o, r, terminal, info, copy(env))
end

function save_gif(env::EnvState, filename::String="default.mp4")
    framerate = Int(1.0/env.params.dt) * 2
    frames = Reel.Frames(MIME("image/png"), fps=framerate)

    cam = CarFollowCamera(EGO_ID, 20.0)
    ego = get_by_id(env.ego, EGO_ID)

    ticks = nframes(env.rec)
    for frame_index in 1:ticks
        scene = env.rec[frame_index-ticks]
        ego = scene[findfirst(EGO_ID, scene)]

        overlays = [TextOverlay(text=["$(veh.id)"], incameraframe=true,
                            pos=VecE2(veh.state.posG.x-0.7, veh.state.posG.y+0.7)) for veh in scene]

        action_state = reshape(env.action_state, 4, :)'
        action_state = action_state[frame_index, :]
        jerk_text = @sprintf("Jerk:  %2.2f m/s^3", action_state[1])
        δrate_text = @sprintf("δ rate:  %2.2f rad/s", action_state[2])
        acc_text = @sprintf("acc:  %2.2f m/s^2", action_state[3])
        δ_text = @sprintf("δ:  %2.2f rad", action_state[4])
        v_text = @sprintf("v:  %2.2f m/s", ego.state.v)
        lane_text = @sprintf("Target Lane: LaneTag(%d, %d)", env.init_lane.segment, env.init_lane.lane)
        action_overlay = TextOverlay(text=[jerk_text, δrate_text,
                            acc_text, δ_text, v_text, lane_text], font_size=14)
        push!(overlays, action_overlay)

        push!(frames, render(scene, env.roadway, overlays, cam=cam, car_colors=env.colours))
    end
    Reel.write(filename, frames)

    write_data(env, replace(filename, "mp4" => "dat"))
end

function write_data(env::EnvState, filename::String="default.dat")
    if filename == ".dat"
        return
    end

    open(filename, "w") do f
        merge_tick = env.merge_tick
        steps = env.steps
        write(f, "$merge_tick\n")
        write(f, "$steps\n")

        for field in env.ego_data
            key = field[1]
            val = field[2]
            val = reshape(val, (1, length(val)))
            write(f, "$key,")
            writedlm(f, val, ",")
        end

        if length(env.car_data) > 1
            write(f, "NONE\n")
        elseif !isnothing(env.victim_id)
            victim = env.car_data[env.victim_id]
            for field in victim
                key = "victim_" * field[1]
                val = field[2]
                val = reshape(val, (1, length(val)))
                write(f, "$key,")
                writedlm(f, val, ",")
            end
        end
    end
end

end # module
