module TCNData

using AutomotiveDrivingModels
using AutoViz
using StatsBase
using Reel
using Printf

export reset, step!, save_gif
export TCNEnv

include("../agent/agent.jl")
include("./structures.jl")
include("./helpers.jl")
include("../behaviours/mpc_driver.jl")

function make_env(params::TCNParams)
    if params.stadium
        roadway = gen_stadium_roadway(params.lanes, length=params.length,
                                                        width=0.0, radius=10.0)
    else
        roadway = gen_straight_roadway(params.lanes, params.length)
    end

    scene, models, colours = populate_others(params, roadway, -1)
    rec = SceneRecord(params.max_ticks, params.dt)

    car_ids = sample(1:params.cars, params.sampled, replace=false)
    for id in car_ids
        colours[id] = COLOR_CAR_EGO
    end

    neighbours = Dict()
    features = Dict()
    D = zeros(params.sampled * params.max_neighbours * 2, params.features)

    TCNEnv(params, roadway, scene, rec, models, colours,
            car_ids, neighbours, features, D)
end

function observe(env::TCNEnv)
    # update neighbours and features
    # return a matrix of dimensions (2*env.params.sampled, feature dim)

    if isempty(env.prev_neighbours) && isempty(env.prev_features)
        for car_id in env.car_ids
            env.prev_neighbours[car_id] = get_neighbours(env, car_id,
                                                                    frenet=true)
            env.prev_features[car_id] = get_features(env, car_id,
                                                    env.prev_neighbours[car_id])
            display(env.prev_features[car_id])
        end

        return env, 0
    else
        for (i, car_id) in enumerate(env.car_ids)
            env.DATA = zeros(env.params.sampled * env.params.max_neighbours * 2,
                                                            env.params.features)

            offsetX = ((i-1) * env.params.max_neighbours * 2) + 1
            offsetY = ((i-1) * env.params.max_neighbours * 2) + 1 +
                                                    env.params.max_neighbours

            num_neighbours = length(env.prev_neighbours[car_id])
            env.DATA[offsetX:offsetY-1, :] =
                vcat(env.prev_features[car_id],
                    zeros(env.params.max_neighbours - num_neighbours,
                                                        env.params.features))

            new_neighbours = get_neighbours(env, car_id, frenet=true)
            new_features = get_features(env, car_id, new_neighbours)

            still_neighbours = filter(c -> c in env.prev_neighbours[car_id],
                                                                new_neighbours)
            idx_prev = indexin(still_neighbours, env.prev_neighbours[car_id])
            idx_new = indexin(still_neighbours, new_neighbours)

            env.DATA[offsetY .+ idx_prev .- 1, :] = new_features[idx_new, :]
            env.prev_neighbours[car_id] = new_neighbours
            env.prev_features[car_id] = new_features
        end

        return env, copy(env.DATA)
    end

    return env, 0
end

function burn_in_sim!(env::TCNEnv; steps::Int=100)
    actions = Array{Any}(undef, length(env.scene))
    for step in 1:steps
        get_actions!(actions, env.scene, env.roadway, env.car_models)
        tick!(env, actions)
    end

    env
end

function Base.reset(paramdict::Dict)
    params = dict_to_tcnparams(paramdict)
    env = make_env(params)
    env = burn_in_sim!(env)
    while is_terminal(env)
        env = make_env(params)
        env = burn_in_sim!(env)
    end
    update!(env.rec, env.scene)

    env, o = observe(env)

    (env, o)
end

function is_terminal(env::TCNEnv; init::Bool=false)
    done = false

    for (i, veh) in enumerate(env.scene)
        if veh.id in env.car_ids
            road_proj = proj(veh.state.posG, env.roadway)
            # off roadway
            done = done || (abs(road_proj.curveproj.t) > DEFAULT_LANE_WIDTH/2.0)
            if done
                break
            end
        end
    end

    done = done || is_crash(env, init=true)

    done
end

function AutomotiveDrivingModels.tick!(env::TCNEnv, actions::Vector{Any})
    for (i, veh) in enumerate(env.scene)
        state′ = propagate(veh, actions[i], env.roadway, env.params.dt)
        env.scene[veh.id] = Entity(state′, veh.def, veh.id)
    end

    env
end

function AutomotiveDrivingModels.get_actions!(
    actions::Vector{A},
    scene::EntityFrame{S, D, I},
    roadway::R,
    models::Dict{I, M},) where {S, D, I, A, R, M <: DriverModel}
    for (i, veh) in enumerate(scene)
        model = models[veh.id]
        AutomotiveDrivingModels.observe!(model, scene, roadway, veh.id)
        actions[i] = rand(model)
    end

    actions
end

function step!(env::TCNEnv)
    car_actions = Array{Any}(undef, length(env.scene))
    get_actions!(car_actions, env.scene, env.roadway, env.car_models)

    env = tick!(env, car_actions) # move to next state
    update!(env.rec, env.scene)
    r = 1.0
    env, o = observe(env)
    terminal = is_terminal(env)

    (o, r, terminal, [], copy(env))
end

function save_gif(env::TCNEnv, filename::String="default.gif")
    framerate = Int(1.0/env.params.dt)
    frames = Reel.Frames(MIME("image/png"), fps=framerate)

    cam = FitToContentCamera(0.01)

    ticks = nframes(env.rec)
    for frame_index in 1:ticks
        scene = env.rec[frame_index-ticks]
        push!(frames, render(scene, env.roadway,
                                cam=cam, car_colors=env.car_colours))
    end
    Reel.write(filename, frames)
end

end # module
