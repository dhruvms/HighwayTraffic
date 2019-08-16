using Random

"""
    get_lane(roadway::Roadway, vehicle::Vehicle)
    get_lane(roadway::Roadway, vehicle::VehicleState)
return the lane where `vehicle` is in.
"""
function get_lane(roadway::Roadway, vehicle::Vehicle)
    lane_tag = vehicle.state.posF.roadind.tag
    return roadway[lane_tag]
end
function get_lane(roadway::Roadway, vehicle::VehicleState)
    lane_tag = vehicle.posF.roadind.tag
    return roadway[lane_tag]
end

"""
    get_end(lane::Lane)
return the end longitudinal position of a lane
"""
function get_end(lane::Lane)
    s_end = round(lane.curve[end].s)
    return s_end
end

"""
    distance_from_end(params::EnvParams, veh::Agent)
returns the distance remaining before the vehicle reaches the end of the
roadway
"""
function distance_from_end(params::EnvParams, veh::Agent)
    return (params.length - veh.state.state.posF.s) / params.length
end

function dict_to_simparams(params::Dict)
    seed = get(params, "seed", 68845)
    # Random.seed!(seed)

    length = get(params, "length", 1000.0)
    lanes = get(params, "lanes", 3)
    cars = get(params, "cars", 30)
    dt = get(params, "dt", 0.2)
    max_ticks = get(params, "max_steps", 256)
    stadium = get(params, "stadium", false)
    change = get(params, "change", false)
    both = get(params, "both", false)
    fov = get(params, "fov", 50)
    beta = get(params, "beta_dist", false)
    clamp = get(params, "clamp_in_sim", false)
    extra_deadends = get(params, "extra_deadends", false)
    eval = get(params, "eval", false)
    norm_obs = get(params, "norm_obs", true)

    cars_per_lane = Int(ceil(cars/lanes))
    room = CAR_LENGTH * 1.1
    room = stadium ? room / 2.0 : room
    rooms = zeros(lanes, cars_per_lane)
    for l in 1:lanes
        rooms[l, :] = cumsum(((0.6 * rand(cars_per_lane)) .+ 1) * room)
    end

    v_des = get(params, "v_des", 5.0)
    ego_dim = get(params, "ego_dim", 9)
    other_dim = get(params, "other_dim", 7)
    occupancy = get(params, "occupancy", false)

    ego_pos = rand(1:cars)
    o_dim = nothing
    if lanes == 1
        o_dim = ego_dim + (2 * other_dim) * (cars > 1)
    elseif lanes == 2
        o_dim = ego_dim + (4 * other_dim) * (cars > 1)
    else
        o_dim = ego_dim + (6 * other_dim) * (cars > 1)
    end

    j_cost = get(params, "j_cost", 0.001)
    δdot_cost = get(params, "d_cost", 0.01)
    a_cost = get(params, "a_cost", 0.1)
    v_cost = get(params, "v_cost", 2.5)
    ϕ_cost = get(params, "phi_cost", 1.0)
    t_cost = get(params, "t_cost", 10.0)
    deadend_cost = get(params, "end_cost", 1.0)

    # # costs = [j_cost, δdot_cost, a_cost, v_cost, ϕ_cost, t_cost, deadend_cost]
    # costs = [v_cost, ϕ_cost, t_cost, deadend_cost]
    # costs = costs ./ sum(costs)
    # # j_cost, δdot_cost, a_cost, v_cost, ϕ_cost, t_cost, deadend_cost = costs
    # v_cost, ϕ_cost, t_cost, deadend_cost = costs

    EnvParams(length, lanes, cars, dt, max_ticks, rooms, stadium, change, both,
                fov, beta, clamp, extra_deadends, eval, norm_obs,
                ego_pos, v_des, ego_dim, other_dim, o_dim, occupancy,
                j_cost, δdot_cost, a_cost, v_cost, ϕ_cost, t_cost, deadend_cost)
end

function get_initial_egostate(params::EnvParams, roadway::Roadway{Float64})
    if params.stadium
        segment = (params.ego_pos % 2) * 6 + (1 - params.ego_pos % 2) * 3
        lane = params.lanes -
                        ((floor((params.ego_pos + 1) / 2) - 1) % params.lanes)
    else
        segment = 1
        lane = params.lanes - (params.ego_pos % params.lanes)
    end
    segment = Int(segment)
    lane = Int(lane)

    lane0 = LaneTag(segment, lane)
    s0 = params.rooms[lane, Int(ceil(params.ego_pos/params.lanes))]
    v0 = rand() + 1.0

    # gap = try
    #     params.rooms[lane, Int(ceil(params.ego_pos/params.lanes))+1] - s0 -
    #                                                                 CAR_LENGTH
    # catch
    #     Inf
    # end
    # t_stop = 1.5
    # stop_dist = v0 * t_stop - 0.5 * 2.0 * t_stop^2
    # while stop_dist > gap
    #     v0 *= 0.9
    #     stop_dist = v0 * t_stop - 0.5 * 2.0 * t_stop^2
    # end

    t0 = (rand() - 0.5) * (DEFAULT_LANE_WIDTH/2.0)
    ϕ0 = (2 * rand() - 1) * 0.1
    # t0 = 0.0
    # ϕ0 = 0.0
    ego = Entity(AgentState(roadway, v=v0, s=s0, t=t0, ϕ=ϕ0, lane=lane0),
                                                        EgoVehicle(), EGO_ID)
    # ego = Entity(AgentState(roadway, v=0.0, s=s0, t=t0, ϕ=ϕ0, lane=lane0),
    #                                                     EgoVehicle(), EGO_ID)
    return Frame([ego]), lane0
end

function populate_scene(params::P, roadway::Roadway{Float64},
                            ego::Entity{AgentState,EgoVehicle,Int64},
                            ) where P <: AbstractParams
    scene = Scene()
    carcolours = Dict{Int, Colorant}()
    models = Dict{Int, DriverModel}()

    push!(scene, Vehicle(ego))
    carcolours[EGO_ID] = COLOR_CAR_EGO

    room = CAR_LENGTH * 1.1
    ego_lane = Int(params.lanes - (params.ego_pos % params.lanes))
    ego_s = params.rooms[ego_lane, Int(ceil(params.ego_pos/params.lanes))]
    s_deadend = ego_s + (25.0 * rand() + 5.0)
    ignore_idx = findall(x -> x,
                    abs.(params.rooms[ego_lane, :] .- s_deadend) .< room)
    lane_deadend = get_lane(roadway, ego.state.state)
    posF = Frenet(roadway[lane_deadend.tag], s_deadend, 0.0, 0.0)
    push!(scene, Vehicle(VehicleState(posF, roadway, 0.0),
                                                    VehicleDef(), 101))
    models[101] = ProportionalSpeedTracker()
    carcolours[101] = RGB(0, 0, 0)
    AutomotiveDrivingModels.set_desired_speed!(models[101], 0.0)

    ignored_cars = Dict(ego_lane => ignore_idx)

    if params.extra_deadends
        extra_deadends = rand(0:params.lanes-1)
        idx = 102
        lanes = [ego_lane]
        for d in 1:extra_deadends
            lane = Int(rand(filter(l->l ∉ lanes, 1:params.lanes)))
            s = rand(params.rooms[lane, :]) + (100.0 * rand() + 100.0)
            ignore_idx = findall(x -> x,
                                abs.(params.rooms[lane, :] .- s) .< room)
            ignored_cars[lane] = ignore_idx

            ego_tag = get_lane(roadway, ego.state.state)
            deadend_tag = LaneTag(ego_tag.tag.segment, lane)
            posF = Frenet(roadway[deadend_tag], s, 0.0, 0.0)
            push!(scene, Vehicle(VehicleState(posF, roadway, 0.0),
                                                            VehicleDef(), idx))
            models[idx] = ProportionalSpeedTracker()
            carcolours[idx] = RGB(0, 0, 0)
            AutomotiveDrivingModels.set_desired_speed!(models[idx], 0.0)

            idx += 1
            push!(lanes, lane)
        end
    end

    v_num = params.ego_pos == -1 ? 1 : EGO_ID + 1
    for i in 1:(params.cars)
        if i == params.ego_pos
            continue
        end

        if params.stadium
            segment = (i % 2) * 6 + (1 - i % 2) * 3
            lane = params.lanes - ((floor((i + 1) / 2) - 1) % params.lanes)
        else
            segment = 1
            lane = params.lanes - (i % params.lanes)
        end
        segment = Int(segment)
        lane = Int(lane)

        if lane ∈ keys(ignored_cars) &&
                Int(ceil(i/params.lanes)) ∈ ignored_cars[lane]
            continue
        end

        type = rand()

        lane0 = LaneTag(segment, lane)
        s0 = params.rooms[lane, Int(ceil(i/params.lanes))]
        v0 = rand() + 1.0

        # gap = try
        #     params.rooms[lane, Int(ceil(i/params.lanes))+1] - s0 - CAR_LENGTH
        # catch
        #     Inf
        # end
        # t_stop = 1.5
        # stop_dist = v0 * t_stop - 0.5 * 2.0 * t_stop^2
        # while stop_dist > gap
        #     v0 *= 0.9
        #     stop_dist = v0 * t_stop - 0.5 * 2.0 * t_stop^2
        # end

        t0 = (rand() - 0.5) * (DEFAULT_LANE_WIDTH/2.0)
        ϕ0 = (2 * rand() - 1) * 0.1
        # t0 = 0.0
        # ϕ0 = 0.0
        posF = Frenet(roadway[lane0], s0, t0, ϕ0)

        push!(scene, Vehicle(VehicleState(posF, roadway, v0),
                                                        VehicleDef(), v_num))
        # push!(scene, Vehicle(VehicleState(posF, roadway, 0.0),
        #                                                 VehicleDef(), v_num))
        if type < 0.0
            models[v_num] = MPCDriver(params.dt)
            v0 = 0.0
            carcolours[v_num] = try
                MONOKAI["color3"]
            catch
                MONOKAY["color3"]
            end
        else
            η_coop = rand()
            η_percept = (rand() - 0.5) * (0.15/0.5)
            models[v_num] = BafflingDriver(params.dt,
                                    η_coop=η_coop,
                                    η_percept=η_percept,
                                    mlon=BafflingLongitudinalTracker(
                                        δ=rand()+3.5,
                                        T=rand()+1.0,
                                        s_min=(rand()*2.0)+1.0,
                                        a_max=rand()+2.5,
                                        d_cmf=rand()+1.5,
                                        ΔT=params.dt,
                                        ),
                                    )
            carcolours[v_num] = HSV(0, 1.0 - η_coop, 1.0)
        end
        v_des = (rand() * 3.0) + 2.0
        AutomotiveDrivingModels.set_desired_speed!(models[v_num], v_des)
        v_num += 1
    end

    (scene, models, carcolours)
end

function get_neighbours(env::EnvState, ego_idx::Int)
    fore_M = get_neighbor_fore_along_lane(env.scene, ego_idx, env.roadway,
                VehicleTargetPointRear(), VehicleTargetPointRear(),
                VehicleTargetPointRear(), max_distance_fore=env.params.length)
    fore_L = get_neighbor_fore_along_left_lane(env.scene, ego_idx, env.roadway,
                VehicleTargetPointRear(), VehicleTargetPointRear(),
                VehicleTargetPointRear(), max_distance_fore=env.params.length)
    fore_R = get_neighbor_fore_along_right_lane(env.scene, ego_idx, env.roadway,
                VehicleTargetPointRear(), VehicleTargetPointRear(),
                VehicleTargetPointRear(), max_distance_fore=env.params.length)
    rear_M = get_neighbor_rear_along_lane(env.scene, ego_idx, env.roadway,
                VehicleTargetPointFront(), VehicleTargetPointFront(),
                VehicleTargetPointFront(), max_distance_rear=env.params.length)
    rear_L = get_neighbor_rear_along_left_lane(env.scene, ego_idx, env.roadway,
                VehicleTargetPointFront(), VehicleTargetPointFront(),
                VehicleTargetPointFront(), max_distance_rear=env.params.length)
    rear_R = get_neighbor_rear_along_right_lane(env.scene, ego_idx, env.roadway,
                VehicleTargetPointFront(), VehicleTargetPointFront(),
                VehicleTargetPointFront(), max_distance_rear=env.params.length)

    (fore_M, fore_L, fore_R, rear_M, rear_L, rear_R)
end

function get_featurevec(env::EnvState, neighbour::NeighborLongitudinalResult,
                            ego_lanetag::LaneTag;
                            lane::Int=0, rear::Bool=false)
    if isnothing(neighbour.ind) ||
            abs(neighbour.Δs) > 10 * CAR_LENGTH || # car is too far
                neighbour.ind >= 101 # neighbour is deadend car
        return zeros(env.params.other_dim)
    else
        veh = env.scene[neighbour.ind]
        veh_proj = proj(veh.state.posG, env.roadway)
        if (abs(veh_proj.curveproj.t) > DEFAULT_LANE_WIDTH/2.0)
            # neighbour is off road
            return zeros(env.params.other_dim)
        end

        ego = get_by_id(env.ego, EGO_ID)

        Δs = (neighbour.Δs * (rear * -1 + !rear * 1))
        Δv = env.scene[neighbour.ind].state.v - ego.state.state.v
        lane_id = zeros(3)
        lane_id[lane] = 1
        neighbour_proj = Frenet(env.scene[neighbour.ind].state.posG,
                                        env.roadway[ego_lanetag], env.roadway)
        Δt = neighbour_proj.t - ego.state.state.posF.t
        Δϕ = WrapPosNegPi(neighbour_proj.ϕ - ego.state.state.posF.ϕ)
        if abs(Δϕ) > 2π/3 # car is turned around
            return zeros(env.params.other_dim)
        end

        vec = vcat([Δs, Δt, Δϕ, Δv], lane_id)
        return vec
    end
end

function get_neighbour_featurevecs(env::EnvState)
    ego_idx = findfirst(EGO_ID, env.scene)
    ego = get_by_id(env.ego, EGO_ID)
    ego_lane = get_lane(env.roadway, ego.state.state)

    fore_M, fore_L, fore_R, rear_M, rear_L, rear_R = get_neighbours(env,
                                                                    ego_idx)

    # lane = 1 (left), 2 (middle), or 3 (right)
    fore_M = get_featurevec(env, fore_M, ego_lane.tag, lane=2)
    fore_L = get_featurevec(env, fore_L, ego_lane.tag, lane=1)
    fore_R = get_featurevec(env, fore_R, ego_lane.tag, lane=3)
    rear_M = get_featurevec(env, rear_M, ego_lane.tag, lane=2, rear=true)
    rear_L = get_featurevec(env, rear_L, ego_lane.tag, lane=1, rear=true)
    rear_R = get_featurevec(env, rear_R, ego_lane.tag, lane=3, rear=true)

    features = nothing
    if env.params.lanes == 1
        features = vcat(fore_M, rear_M)
    elseif env.params.lanes == 2
        features = vcat(fore_M, fore_L, rear_M, rear_L)
    else
        features = vcat(fore_M, fore_L, fore_R, rear_M, rear_L, rear_R)
    end

    features
end

function get_ego_features(env::EnvState)
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

    deadend = env.scene[findfirst(101, env.scene)]
    deadend_s = (deadend.state.posF.s - ego.state.state.posF.s) /
                                                            deadend.state.posF.s
    deadend_s = max(deadend_s, 0.0)

    # TODO: normalise?
    ego_o = [deadend_s, in_lane, t, ϕ, v, a, δ, action[1], action[2]]

    ego_o
end

function relative_lane(l_ego::Int, l_other::Int)
    if l_ego == l_other
        return 2
    elseif l_ego > l_other
        return 3
    else
        return 1
    end
end

function map_to_range(in::Float64, in_start::Float64, in_end::Float64,
                                        out_start::Float64, out_end::Float64)
    out = out_start +
            ((out_end - out_start) / (in_end - in_start)) * (in - in_start)
    out
end

function map_to_01(in::Float64, in_start::Float64, in_end::Float64)
    return map_to_range(in, in_start, in_end, 0.0, 1.0)
end

function get_occupancy_image(env::EnvState)
    fov = 2 * env.params.fov + 1
    ego = get_by_id(env.ego, EGO_ID)
    ego_lane = get_lane(env.roadway, ego.state.state)
    ego_r_start = Int(env.params.fov + 2 - ego.def.length/2.0)
    ego_r_end = Int(env.params.fov + 1 + ego.def.length/2.0)
    ego_rows = ego_r_start:ego_r_end

    occupancy = zeros(fov, 3)
    rel_vel = zeros(fov, 3)
    rel_lat_disp = zeros(fov, 3)
    rel_heading = zeros(fov, 3)

    for (i,veh) in enumerate(env.scene)
        if veh.id != EGO_ID
            veh_proj = Frenet(veh.state.posG,
                                env.roadway[ego_lane.tag],
                                env.roadway)
            Δs = veh_proj.s - ego.state.state.posF.s
            Δlane = abs(ego_lane.tag.lane - veh.state.posF.roadind.tag.lane)
            if abs(Δs) < env.params.fov && Δlane ≤ 1 # veh is in range
                Δv = veh.state.v - ego.state.state.v
                Δt = abs(veh_proj.t - ego.state.state.posF.t)
                Δϕ = Wrap2Pi(veh_proj.ϕ - ego.state.state.posF.ϕ)
                lane = relative_lane(ego_lane.tag.lane,
                                        veh.state.posF.roadind.tag.lane)

                if Δs >= 0
                    veh_row = env.params.fov + 1 - floor(Δs)
                    r_start = max(Int(veh_row - (veh.def.length/2.0 - 1.0)), 1)
                    r_end = min(Int(veh_row + veh.def.length/2.0), fov)
                    veh_rows = r_start:r_end
                    if veh.id >= 101
                        veh_rows = 1:r_end
                    end
                else
                    veh_row = env.params.fov + 1 - ceil(Δs)
                    r_start = max(Int(veh_row - (veh.def.length/2.0 - 1.0)), 1)
                    r_end = min(Int(veh_row + veh.def.length/2.0), fov)
                    veh_rows = r_start:r_end
                    if veh.id >= 101
                        veh_rows = r_start:fov
                    end
                end

                occupancy[veh_rows, lane] .= 1
                if veh.id <= 100
                    if env.params.norm_obs
                        rel_vel[veh_rows, lane] .= map_to_01(Δv, -env.params.v_des,
                                                                env.params.v_des)
                        rel_lat_disp[veh_rows, lane] .= Δt / DEFAULT_LANE_WIDTH
                        rel_heading[veh_rows, lane] .= Δϕ / 2π
                    else
                        rel_vel[veh_rows, lane] .= Δv
                        rel_lat_disp[veh_rows, lane] .= Δt
                        rel_heading[veh_rows, lane] .= Δϕ
                    end
                else
                    rel_vel[veh_rows, lane] .= 0.0
                    rel_lat_disp[veh_rows, lane] .= 0.0
                    rel_heading[veh_rows, lane] .= 0.0
                end
            end
        end
    end

    occupancy[ego_rows, 2] .= 1
    rel_vel[ego_rows, 2] .= ego.state.state.v
    rel_lat_disp[ego_rows, 2] .= ego.state.state.posF.t
    rel_heading[ego_rows, 2] .= ego.state.state.posF.ϕ

    # vcat(occupancy, rel_vel, rel_lat_disp, rel_heading)
    result = cat(occupancy, rel_vel, rel_lat_disp, rel_heading, dims=3)
    # result = permutedims(result, [3, 1, 2])
    result
end

function is_crash(env::E; init::Bool=false) where E <: AbstractEnv
    # ego = env.scene[findfirst(EGO_ID, env.scene)]
    ego =   try
                Vehicle(get_by_id(env.ego, EGO_ID))
            catch
                nothing
            end

    if !init
        for veh in env.scene
            if veh.id != EGO_ID
                if !isnothing(ego) && is_colliding(ego, veh)
                    return true
                end
            end
        end
    else
        for i in 1:length(env.scene)-1
            for j in i+1:length(env.scene)
                if is_colliding(env.scene[i], env.scene[j])
                    return true
                end
            end
        end
    end

    return false
end
