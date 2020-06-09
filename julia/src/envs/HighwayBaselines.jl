module HighwayBaselines

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
        	# change to any other lane
            lane = try
                rand(filter(l->l != lanetag.lane, 1:params.lanes))
            catch
                1
            end
        else
        	# change to immediate left lane
            lane = lane + 1
        end
    elseif params.both
    	# change or follow
        lane = rand(1:params.lanes)
    end
    lanetag = LaneTag(seg, lane)
    veh = get_by_id(ego, EGO_ID)
    ego_proj = Frenet(veh.state.state.posG, roadway[lanetag], roadway)

    action_state = []
    ego_data = Dict{String, Vector{Float64}}(
                    "deviation" => [abs(ego_proj.t)],
                    "vel" => [veh.state.state.v],
                    "alat" => [0.0],
                    "alon" => [0.0])
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
    if !isnothing(params.ego_model)
    	if params.ego_model == 1
    		# IDM+MOBIL
    		ego_model = Tim2DDriver(
    							params.dt,
								mlon=IntelligentDriverModel(
										s_min=(rand()*2.0)+1.0,
                                        a_max=rand()+2.5,
                                        d_cmf=rand()+1.5,
                                        ΔT=params.dt,
                                        ),
								mlat=ProportionalLaneTracker(),
								mlane=MOBIL(
										params.dt,
										mlon=IntelligentDriverModel(
												s_min=(rand()*2.0)+1.0,
		                                        a_max=rand()+2.5,
		                                        d_cmf=rand()+1.5,
		                                        ΔT=params.dt,
		                                        ),
										safe_decel=rand()+1.5,
										),
	                            )
    	else
    		# MPC
    		ego_model = MPCDriver(params.dt, weight=false)
    	end
    end

    EnvState(params, roadway, scene, rec, ego, action_state, lanetag, steps,
                mpc,
                in_lane, lane_ticks, victim_id,
                merge_tick, min_dist, lane_dist, car_data, ego_data,
                models, colours, prev_shaping,
                ego_model)
end

"""
    burn_in_sim!(env::EnvState; steps::Int=0)
burn in simulation state by propagating all other vehicles for some timesteps
"""
function burn_in_sim!(env::EnvState; steps::Int=0)
    actions = Array{Any}(undef, length(env.scene))
    for step in 1:steps
        get_actions!(actions, env.scene, env.roadway,
        				env.ego_model, env.other_cars)
        tick!(env, actions, init=true)
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
    env.min_dist = min(env.min_dist, min_dist)
    update!(env.rec, env.scene)

    env
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

    ego = env.scene[findfirst(EGO_ID, env.scene)]
    road_proj = proj(ego.state.posG, env.roadway)

    min_dist, crash = is_crash(env, init=init, baseline=true)
    if (abs(road_proj.curveproj.t) > DEFAULT_LANE_WIDTH/2.0) || # off roadway
							crash || # collision
							env.steps ≥ env.params.max_ticks # timeout
        done = true
        final_r = -100.0
    end

    # done = done || (env.steps ≥ env.params.max_ticks)
    if (env.lane_ticks ≥ 10)
        done = true
        final_r = +100.0
    end

    # if crash
    # 	println("CRASH")
    # elseif abs(road_proj.curveproj.t) > DEFAULT_LANE_WIDTH/2.0
    # 	println("OFF ROAD")
    # elseif env.steps ≥ env.params.max_ticks
    # 	println("TIMEOUT")
    # elseif done
    # 	println("SUCCESS!")
    # end

    done, final_r, min_dist
end

"""
    reward(env::EnvState, action::Vector{Float64},
                    deadend::Float64, in_lane::Bool)
calculate reward
"""
function reward(env::EnvState, action::Vector{Float64})
    ego = env.scene[findfirst(EGO_ID, env.scene)]
    lane = get_lane(env.roadway, ego.state)
    in_lane = lane.tag.lane == env.init_lane.lane ? true : false

    deadend = env.scene[findfirst(101, env.scene)]
    deadend = (deadend.state.posF.s - ego.state.posF.s) /
                                                        deadend.state.posF.s
    deadend = max(deadend, 0.0)

    true_lanetag = LaneTag(lane.tag.segment, env.init_lane.lane)
    ego_proj = Frenet(ego.state.posG,
                        env.roadway[true_lanetag], env.roadway)

    shaping = 0.0
    # # acceleration cost
    # shaping -= env.params.a_cost * abs(ego.state.a)
    # desired velocity cost
    shaping -= env.params.v_cost * abs(ego.state.v - env.params.v_des)
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
            env.lane_ticks += 1
        else
            env.in_lane = true
            env.lane_ticks = 1
        end

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
    reward -= env.params.a_cost * sum(abs.(action))

    (env, reward)
end

"""
    AutomotiveDrivingModels.tick!(env::EnvState, action::Vector{Float64},
                                        actions::Vector{Any}; init::Bool=false)
propagate all vehicles one timestep ahead
"""
function AutomotiveDrivingModels.tick!(env::EnvState,
                                        actions::Vector{Any}; init::Bool=false)
	ego_action = nothing

    for (i, veh) in enumerate(env.scene)
        if veh.id == EGO_ID
            if !init
				ego_action = [actions[i].a_lat, actions[i].a_lon]

            	state′ = propagate(veh, actions[i], env.roadway, env.params.dt)
            	env.scene[findfirst(EGO_ID, env.scene)] = Entity(state′, veh.def, veh.id)

                lane = get_lane(env.roadway, veh.state)

                true_lanetag = LaneTag(lane.tag.segment, env.init_lane.lane)
                ego_proj = Frenet(veh.state.posG,
                                    env.roadway[true_lanetag], env.roadway)

                push!(env.ego_data["deviation"], abs(ego_proj.t))
                push!(env.ego_data["vel"], state′.v)
                push!(env.ego_data["alat"], actions[i].a_lat)
                push!(env.ego_data["alon"], actions[i].a_lon)
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

    (env, ego_action)
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
    ego_model::M,
    models::Dict{I, M}, # id → model
    init_lane::Int64=2,
    dir::Union{Int, Nothing}=nothing,
    mpc::Bool=false,
    mpc_s::Union{Int, Nothing}=nothing,
    mpc_cf::Union{Float64, Nothing}=nothing,
    mpc_cm::Union{Int, Nothing}=nothing,
    ) where {S, D, I, A, R, M <: DriverModel}

    for (i, veh) in enumerate(scene)
        if veh.id == EGO_ID
        	if mpc
        		ego = scene[findfirst(EGO_ID, scene)]
			    lane = get_lane(roadway, ego.state)
			    true_lanetag = LaneTag(lane.tag.segment, init_lane)
			    ego_proj = Frenet(ego.state.posG,
			                        roadway[true_lanetag], roadway)
			    target_roadind = move_along(ego_proj.roadind, roadway,
			                                CAR_LENGTH * mpc_s)
			    goal = Frenet(target_roadind, roadway)
			    AutomotiveDrivingModels.observe!(
			    			ego_model, scene, roadway, EGO_ID,
			    			ego_proj, ego.state.v, goal, mpc_cf, mpc_cm)
        	else
				AutomotiveDrivingModels.observe!(ego_model, scene, roadway, EGO_ID, dir)
			end

			actions[i] = rand(ego_model)
        elseif veh.id >= 101
            actions[i] = LatLonAccel(0, 0)
        else
	        model = models[veh.id]
	        AutomotiveDrivingModels.observe!(model, scene, roadway, veh.id)
	        actions[i] = rand(model)
        end
    end

    actions
end

"""
    step!(env::EnvState, action::Vector{Float32})
take one action in the environment
"""
function step!(env::EnvState)
	ego = env.scene[findfirst(EGO_ID, env.scene)]
    lane = get_lane(env.roadway, ego.state)
    in_lane = lane.tag.lane == env.init_lane.lane ? true : false
    dir = in_lane ? DIR_MIDDLE : nothing # DIR_LEFT?
    mpc = env.params.ego_model == 2

    all_actions = Array{Any}(undef, length(env.scene))
    get_actions!(all_actions, env.scene, env.roadway,
    				env.ego_model, env.other_cars,
    				env.init_lane.lane, dir,
    				mpc, env.params.mpc_s, env.params.mpc_cf, env.params.mpc_cm)

    env, ego_action = tick!(env, all_actions) # move to next state
    update!(env.rec, env.scene)
    env.steps += 1

    terminal, final_r, min_dist = is_terminal(env)
    env.min_dist = min(env.min_dist, min_dist)

    env, r = reward(env, ego_action)
    if Bool(terminal)
        r = final_r
    end

    (r, terminal, copy(env))
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

        ticks = nframes(env.rec)
        for frame_index in 1:ticks
            scene = env.rec[frame_index-ticks]
            ego = scene[findfirst(EGO_ID, scene)]
            lane = get_lane(env.roadway, ego.state)
		    true_lanetag = LaneTag(lane.tag.segment, env.init_lane.lane)
		    ego_proj = Frenet(ego.state.posG,
		                        env.roadway[true_lanetag], env.roadway)

            overlays = [TextOverlay(text=["$(veh.id)"], incameraframe=true,
                                pos=VecE2(veh.state.posG.x-0.7, veh.state.posG.y+0.7)) for veh in scene]

	        if env.params.ego_model == 2
		        target_roadind = move_along(ego_proj.roadind, env.roadway,
		                                    CAR_LENGTH * env.params.mpc_s)
		        goal = Frenet(target_roadind, env.roadway)
		        traj, _, _ = get_mpc_trajectory(env.ego_model, scene, env.roadway, EGO_ID,
		                                    ego_proj, ego.state.v, goal)
		        traj_overlay = TrajOverlay(traj)
	        end

            t_text = @sprintf("Abs lateral deviation, t:  %2.2f m", abs(ego_proj.t))
            ϕ_text = @sprintf("Abs heading deviation, ϕ:  %2.2f rad", abs(ego_proj.ϕ))
            v_text = @sprintf("v:  %2.2f m/s", ego.state.v)
            lane_text = @sprintf("Target Lane: LaneTag(%d, %d)", env.init_lane.segment, env.init_lane.lane)
            action_overlay = TextOverlay(text=[t_text, ϕ_text,
                                			v_text, lane_text], font_size=14)
            push!(overlays, action_overlay)

            if env.params.ego_model == 2
            	push!(frames, render(scene, env.roadway, vcat([overlays..., traj_overlay]), cam=cam, car_colors=env.colours))
            else
            	push!(frames, render(scene, env.roadway, overlays, cam=cam, car_colors=env.colours))
            end
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
