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

function dict_to_params(params::Dict)
    length = get(params, "length", 100.0)
    lanes = get(params, "lanes", 2)
    cars = get(params, "cars", 30)
    v_des = get(params, "v_des", 15.0)
    dt = get(params, "dt", 0.2)
    ego_dim = get(params, "ego_dim", 8)
    other_dim = get(params, "other_dim", 7)
    o_dim = ego_dim + 6 * other_dim
    max_ticks = get(params, "max_steps", 7)

    j_cost = get(params, "j_cost", 0.01)
    δdot_cost = get(params, "d_cost", 0.02)
    a_cost = get(params, "a_cost", 0.01)
    v_cost = get(params, "v_cost", 1.0)
    ϕ_cost = get(params, "phi_cost", 1.0)
    t_cost = get(params, "t_cost", 2.0)

    if cars == 1
        ego_dim = 9
    end

    EnvParams(length, lanes, cars, v_des, dt, ego_dim, other_dim, o_dim,
                max_ticks, j_cost, δdot_cost, a_cost, v_cost, ϕ_cost, t_cost)
end

function get_initial_egostate(params::EnvParams, roadway::Roadway{Float64})
    v0 = rand() * params.v_des
    s0 = 0.1
    lane0 = LaneTag(6, rand(1:params.lanes))
    t0 = (DEFAULT_LANE_WIDTH * rand()) - (DEFAULT_LANE_WIDTH/2.0)
    ϕ0 = (2 * rand() - 1) * 0.3 # max steering angle
    ego = Entity(AgentState(roadway, v=v0, s=s0, t=t0, ϕ=ϕ0, lane=lane0), EgoVehicle(), EGO_ID)
    return Frame([ego]), lane0
end

function populate_others(params::EnvParams, roadway::Roadway{Float64})
    scene = Scene()
    carcolours = Dict{Int, Colorant}()
    models = Dict{Int, DriverModel}()

    room = (params.length) / max(1, params.cars-1)
    v_num = EGO_ID + 1
    for i in 1:(params.cars-1)
        seg = (i % 2) * 3 + (1 - i % 2) * 6
        type = rand()

        v0 = rand() * params.v_des
        s0 = i * room
        lane0 = LaneTag(seg, rand(1:params.lanes))
        t0 = 0.0
        ϕ0 = 0.0
        posF = Frenet(roadway[lane0], s0, t0, ϕ0)

        push!(scene, Vehicle(VehicleState(posF, roadway, 0.0), VehicleDef(), v_num))
        if type <= 0.2
            models[v_num] = MPCDriver(params.dt)
            v0 = 0.0
            carcolours[v_num] = MONOKAI["color3"]
        elseif type > 0.2 && type <= 0.6
            models[v_num] = Tim2DDriver(params.dt,
                                        mlon=IntelligentDriverModel(ΔT=params.dt))
            carcolours[v_num] = MONOKAI["color4"]
        else
            models[v_num] = LatLonSeparableDriver( # produces LatLonAccels
                    ProportionalLaneTracker(), # lateral model
                    IntelligentDriverModel(ΔT=params.dt), # longitudinal model
                    )
            carcolours[v_num] = MONOKAI["color5"]
        end
        AutomotiveDrivingModels.set_desired_speed!(models[v_num], v0)
        v_num += 1
    end

    (scene, models, carcolours)
end

function get_neighbours(env::EnvState, ego_idx::Int)
    fore_M = get_neighbor_fore_along_lane(env.scene, ego_idx, env.roadway, VehicleTargetPointFront(), VehicleTargetPointRear(), VehicleTargetPointFront(), max_distance_fore=env.params.length)
    fore_L = get_neighbor_fore_along_left_lane(env.scene, ego_idx, env.roadway, VehicleTargetPointRear(), VehicleTargetPointRear(), VehicleTargetPointFront(), max_distance_fore=env.params.length)
    fore_R = get_neighbor_fore_along_right_lane(env.scene, ego_idx, env.roadway, VehicleTargetPointRear(), VehicleTargetPointRear(), VehicleTargetPointFront(), max_distance_fore=env.params.length)
    rear_M = get_neighbor_rear_along_lane(env.scene, ego_idx, env.roadway, VehicleTargetPointFront(), VehicleTargetPointFront(), VehicleTargetPointRear(), max_distance_rear=env.params.length)
    rear_L = get_neighbor_rear_along_left_lane(env.scene, ego_idx, env.roadway, VehicleTargetPointFront(), VehicleTargetPointFront(), VehicleTargetPointRear(), max_distance_rear=env.params.length)
    rear_R = get_neighbor_rear_along_right_lane(env.scene, ego_idx, env.roadway, VehicleTargetPointFront(), VehicleTargetPointFront(), VehicleTargetPointRear(), max_distance_rear=env.params.length)

    (fore_M, fore_L, fore_R, rear_M, rear_L, rear_R)
end

function get_neighbour_dists(env::EnvState)
    # ego = env.scene[findfirst(EGO_ID, env.scene)]
    ego_idx = findfirst(EGO_ID, env.scene)
    ego = env.scene[findfirst(EGO_ID, env.scene)]

    road_proj = proj(ego.state.posG, env.roadway)

    left_lane_exists = road_proj.tag.lane < env.params.lanes
    right_lane_exists = road_proj.tag.lane > 1
    fore_M, fore_L, fore_R, rear_M, rear_L, rear_R = get_neighbours(env, ego_idx)

    features = [fore_M.Δs, fore_L.Δs, fore_R.Δs,
                rear_M.Δs, rear_L.Δs, rear_R.Δs]
    features /= env.params.length
    if !left_lane_exists
        features[[2, 5]] .= 0.0
    end
    if !right_lane_exists
        features[[3, 6]] .= 0.0
    end
    features
end

function get_featurevec(env::EnvState, neighbour::NeighborLongitudinalResult,
                            ego_proj::RoadProjection{Int64,Float64};
                            lane::Int=0, rear::Bool=false)
    if isnothing(neighbour.ind)
        return zeros(env.params.other_dim)
    else
        Δs = (neighbour.Δs * (rear * -1 + !rear * 1)) / env.params.length
        v = env.scene[neighbour.ind].state.v
        lane_id = zeros(3)
        lane_id[lane] = 1
        neighbour_proj = Frenet(env.scene[neighbour.ind].state.posG, env.roadway[ego_proj.tag], env.roadway)
        Δt = neighbour_proj.t
        Δϕ = neighbour_proj.ϕ

        vec = vcat([Δs, v], lane_id, [Δt, Δϕ])
        return vec
    end
end

function get_neighbour_featurevecs(env::EnvState)
    ego_idx = findfirst(EGO_ID, env.scene)
    ego = env.scene[findfirst(EGO_ID, env.scene)]

    road_proj = proj(ego.state.posG, env.roadway)

    fore_M, fore_L, fore_R, rear_M, rear_L, rear_R = get_neighbours(env, ego_idx)

    # lane = 1 (left), 2 (middle), or 3 (right)
    fore_M = get_featurevec(env, fore_M, road_proj, lane=2)
    fore_L = get_featurevec(env, fore_L, road_proj, lane=1)
    fore_R = get_featurevec(env, fore_R, road_proj, lane=3)
    rear_M = get_featurevec(env, rear_M, road_proj, lane=2, rear=true)
    rear_L = get_featurevec(env, rear_L, road_proj, lane=1, rear=true)
    rear_R = get_featurevec(env, rear_R, road_proj, lane=3, rear=true)

    features = vcat(fore_M, fore_L, fore_R, rear_M, rear_L, rear_R)
    features
end

function is_crash(env::EnvState; init::Bool=false)
    # ego = env.scene[findfirst(EGO_ID, env.scene)]
    ego = get_by_id(env.ego, EGO_ID)

    if ego.state.state.v ≈ 0
        return false
    end

    if !init
        for veh in env.scene
            if veh.id != EGO_ID
                if is_colliding(Vehicle(ego), veh)
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