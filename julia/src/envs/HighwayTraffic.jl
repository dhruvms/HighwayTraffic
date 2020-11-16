module HighwayTraffic

using AutomotiveDrivingModels
using AutoViz
using Reel
using Printf
using DelimitedFiles
using Statistics

export reset, step!, save_gif
export action_space, observation_space, EnvState

include("../agent/agent.jl")
include("./structures.jl")
include("./helpers.jl")
include("../behaviours/mpc_driver.jl")
include("../behaviours/baffling_drivers.jl")
include("./traj_overlay.jl")

"""
    make_env(params::EnvParams)
initialise simulator environment
"""
function make_env(params::EnvParams)
    if params.stadium
        roadway = gen_stadium_roadway(params.lanes, length=params.road, width=0.0, radius=10.0)
    else
        roadway = gen_straight_roadway(params.lanes, params.road)
    end

    ego, lanetag = get_initial_egostate(params, roadway)
    seg = lanetag.segment
    lane = lanetag.lane
    if params.change
        if !params.hri
            lane = try
                rand(filter(l->l != lanetag.lane, 1:params.lanes))
            catch
                1
            end
        else
            lane = lane + 1
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
    merge_tick = -1
    min_dist = Inf
    lane_dist = []
    car_data = Dict{Int, Dict{String, Vector{Float64}}}()
    for (i, veh) in enumerate(scene)
        if veh.id ≠ EGO_ID && veh.id <= 100
            car_data[veh.id] = Dict{String, Vector{Float64}}(
                                    "vel" => [veh.state.v],
                                    "alat" => [0.0],
                                    "alon" => [0.0])
        end
    end

    prev_shaping = nothing
    ego_model = nothing

    EnvState(params, roadway, scene, rec, ego, action_state, lanetag, steps,
                mpc,
                in_lane, lane_ticks, victim_id,
                merge_tick, min_dist, lane_dist, car_data, ego_data,
                models, colours, prev_shaping,
                ego_model)
end

"""
    observe(env::EnvState)
get vector-based observation
"""
function observe(env::EnvState)
    ego_o = get_ego_features(env)
    if env.params.cars - 1 > 0
        other_o = get_neighbour_featurevecs(env)
        o = vcat(ego_o, other_o)
        return o, ego_o[1], ego_o[2]
    end

    return ego_o, ego_o[1], ego_o[2]
end

"""
    observe_occupancy(env::EnvState)
get occupancy grid-based observation
"""
function observe_occupancy(env::EnvState)
    ego_o = get_ego_features(env)
    o = get_occupancy_image(env)

    fov = 2 * env.params.fov + 1
    ego_mat = zeros(fov, 3)
    ego_mat[1:env.params.ego_dim, 1] = ego_o
    o = cat(o, ego_mat, dims=3)

    return o, ego_o[1], ego_o[2]
end

"""
    burn_in_sim!(env::EnvState; steps::Int=0)
burn in simulation state by propagating all other vehicles for some timesteps
"""
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

"""
    Base.reset(paramdict::Dict)
reset environment state
"""
function Base.reset(paramdict::Dict)
    params = dict_to_simparams(paramdict)
    env = make_env(params)
    burn_in_sim!(env)
    check, _, min_dist = is_terminal(env)
    while check
        env = make_env(params)
        burn_in_sim!(env)
        check, _, min_dist = is_terminal(env)
    end
    env.action_state = env.action_state[end-3:end]
    env.min_dist = min(env.min_dist, min_dist)
    update!(env.rec, env.scene)

    if env.params.occupancy
        o, _, _ = observe_occupancy(env)
    else
        o, _, _ = observe(env)
    end

    (env, o)
end

"""
    is_terminal(env::EnvState; init::Bool=false)
check for episode termination
    - success if criteria achieved
    - failure if collision or off-roadway
"""
function is_terminal(env::EnvState; init::Bool=false)
    done = false
    final_r = 0.0

    # ego = env.scene[findfirst(EGO_ID, env.scene)]
    ego = get_by_id(env.ego, EGO_ID)
    road_proj = proj(ego.state.state.posG, env.roadway)

    # done = done || (ego.state.state.v < 0.0) # vehicle has negative velocity
    min_dist, crash = is_crash(env, init=init)
    if (abs(road_proj.curveproj.t) > DEFAULT_LANE_WIDTH/2.0) || # off roadway
                                                                crash
        done = true
        final_r = -100.0
    end

    # done = done || (env.steps ≥ env.params.max_ticks)
    if !env.params.eval # training
        if (env.lane_ticks ≥ 25)
            done = true
            final_r = +100.0
        end
    else # evaluation
        if (env.lane_ticks ≥ 25)
            done = true
            final_r = +100.0
        end
    end

    # max_s = 0.0
    # for (i, veh) in enumerate(env.scene)
    #     if veh.id <= 100 && veh.state.posF.s > max_s
    #         max_s = veh.state.posF.s
    #     end
    # end
    # done = done || (max_s ≥ env.params.road * 0.95)

    done, final_r, min_dist
end

"""
    reward(env::EnvState, action::Vector{Float64},
                    deadend::Float64, in_lane::Bool)
calculate reward
"""
function reward(env::EnvState, action::Vector{Float64},
                    deadend::Float64, in_lane::Bool)
    # ego = env.scene[findfirst(EGO_ID, env.scene)]
    ego = get_by_id(env.ego, EGO_ID)
    lane = get_lane(env.roadway, ego.state.state)

    true_lanetag = LaneTag(lane.tag.segment, env.init_lane.lane)
    ego_proj = Frenet(ego.state.state.posG,
                        env.roadway[true_lanetag], env.roadway)

    shaping = 0.0
    # # acceleration cost
    # shaping -= env.params.a_cost * abs(ego.state.a)
    # desired velocity cost
    shaping -= env.params.v_cost * abs(ego.state.state.v - env.params.v_des)
    # lane follow cost
    Δt = abs(ego_proj.t)
    Δϕ = abs(ego_proj.ϕ)
    shaping -= env.params.t_cost * Δt
    shaping -= env.params.ϕ_cost * Δϕ * in_lane

    # distance to deadend
    if in_lane
        shaping += 1.0
        shaping += env.params.deadend_cost * deadend

        if env.in_lane
            if !env.params.eval # training
                if (Δt ≤ 0.15) && (Δϕ ≤ deg2rad(10))
                    env.lane_ticks += 1
                else
                    env.in_lane = false
                    env.lane_ticks = 0
                end
            else # evaluation
                env.lane_ticks += 1
                # # Dhruv's success check - "well-aligned" with lane
                # if (Δt ≤ 0.15) && (Δϕ ≤ deg2rad(10))
                #     env.lane_ticks += 1
                # else
                #     env.in_lane = false
                #     env.lane_ticks = 0
                # end
            end
        else
            if !env.params.eval # training
                if (Δt ≤ 0.15) && (Δϕ ≤ deg2rad(10))
                    env.in_lane = true
                    env.lane_ticks = 1
                end
            else # evaluation
                env.in_lane = true
                env.lane_ticks = 1
                # # Dhruv's success check - "well-aligned" with lane
                # if (Δt ≤ 0.15) && (Δϕ ≤ deg2rad(10))
                #     env.in_lane = true
                #     env.lane_ticks = 1
                # end
            end
        end

        if env.params.eval
            if env.lane_ticks ≥ 1 && env.merge_tick == -1
                env.merge_tick = env.steps
                victim = get_neighbor_rear_along_lane(
                            env.scene, EGO_ID, env.roadway,
                            VehicleTargetPointFront(), VehicleTargetPointFront(),
                            VehicleTargetPointRear())
                if !isnothing(victim.ind)
                    env.victim_id = victim.ind

                    car_data_copy = copy(env.car_data)
                    for car in keys(car_data_copy)
                        if car ≠ env.victim_id && car ∈ keys(env.car_data)
                            delete!(env.car_data, car)
                        end
                    end
                end
            end
        end
    else
        shaping -= env.params.deadend_cost * (1.0 - deadend)

        if env.in_lane
            env.in_lane = false
            env.lane_ticks = 0
        end
    end

    reward = 0.0
    if !isnothing(env.prev_shaping)
        reward = shaping - env.prev_shaping
    end
    env.prev_shaping = shaping
    push!(env.lane_dist, Δt)

    # action cost
    reward -= env.params.j_cost * abs(action[1])
    reward -= env.params.δdot_cost * abs(action[2])

    (env, reward)
end

"""
    AutomotiveDrivingModels.tick!(env::EnvState, action::Vector{Float64},
                                        actions::Vector{Any}; init::Bool=false)
propagate all vehicles one timestep ahead
"""
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

"""
    AutomotiveDrivingModels.get_actions!(
        actions::Vector{A},
        scene::EntityFrame{S, D, I},
        roadway::R,
        models::Dict{I, M}, # id → model
        ) where {S, D, I, A, R, M <: DriverModel}
get actions for all other vehicles except egovehicle
"""
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

"""
    step!(env::EnvState, action::Vector{Float32})
take one action in the environment
"""
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
    terminal, final_r, min_dist = is_terminal(env)
    env.min_dist = min(env.min_dist, min_dist)
    # if env.params.eval
    #     terminal = false
    # end

    env, r = reward(env, action, deadend, Bool(in_lane))
    r -= 2.0 * neg_v
    if Bool(terminal)
        r = final_r
    end

    ego = get_by_id(env.ego, EGO_ID)
    info = [ego.state.state.posF.s, ego.state.state.posF.t,
                ego.state.state.posF.ϕ, ego.state.state.v]

    (o, r, terminal, info, copy(env))
end

"""
    save_gif(env::EnvState, filename::String="default.mp4")
save video and/or log data for last episode
"""
function save_gif(env::EnvState, filename::String="default.mp4")
    if env.params.video
        framerate = Int(1.0/env.params.dt) * 2
        frames = Reel.Frames(MIME("image/png"), fps=framerate)

        cam = CarFollowCamera(EGO_ID, 8.0)
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
    end

    if env.params.write_data
        write_data(env, replace(filename, "mp4" => "dat"))
    end
end

"""
    write_data(env::EnvState, filename::String="default.dat")
log data for last episode
"""
function write_data(env::EnvState, filename::String="default.dat")
    if filename == ".dat"
        return
    end

    open(filename, "w") do f
        merge_tick = env.merge_tick
        steps = env.steps
        min_dist = env.min_dist
        avg_offset = mean(env.lane_dist)
        write(f, "$merge_tick\n")
        write(f, "$steps\n")
        write(f, "$min_dist\n")
        write(f, "$avg_offset\n")

        for field in env.ego_data
            key = field[1]
            val = field[2]
            val = reshape(val, (1, length(val)))
            write(f, "$key,")
            writedlm(f, val, ",")
        end

        if length(env.car_data) > 1
            write(f, "NONE\n")
            return
        elseif !isnothing(env.victim_id) && env.victim_id < 100
            victim = try
                env.car_data[env.victim_id]
            catch
                write(f, "NONE\n")
                return
            end
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
