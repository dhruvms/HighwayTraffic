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
    cars = get(params, "cars", 1)
    v_des = get(params, "v_des", 10.0)
    dt = get(params, "dt", 0.2)
    o_dim = get(params, "o_dim", 13)

    a_cost = get(params, "a_cost", 0.1)
    δ_cost = get(params, "d_cost", 0.01)
    v_cost = get(params, "v_cost", 0.2)
    ϕ_cost = get(params, "phi_cost", 0.35)
    t_cost = get(params, "t_cost", 0.34)

    num_others = get(params, "num_others", 4)
    random = get(params, "random_others", true)
    stadium = get(params, "stadium", false)

    if num_others == 0
        o_dim = 7
    end

    EnvParams(length, lanes, cars, v_des, dt, o_dim,
                a_cost, δ_cost, v_cost, ϕ_cost, t_cost,
                num_others, random, stadium)
end

function get_initial_egostate(params::EnvParams, roadway::Roadway{Float64})
    v0 = rand() * params.v_des
    s0 = rand() * ((params.length / 2.0) / max(1, params.num_others))
    lane0 = LaneTag(rand(1:(4*params.stadium + 1*!params.stadium)), rand(1:params.lanes))
    t0 = (DEFAULT_LANE_WIDTH * rand()) - (DEFAULT_LANE_WIDTH/2.0)
    ϕ0 = (2 * rand() - 1) * 0.6 # max steering angle
    ego = Entity(AgentState(roadway, v=v0, s=s0, t=t0, ϕ=ϕ0, lane=lane0), EgoVehicle(), EGO_ID)
    return Frame([ego]), lane0
end

function populate_others(params::EnvParams, roadway::Roadway{Float64})
    scene = Scene()
    carcolours = Dict{Int, Colorant}()
    models = Dict{Int, DriverModel}()

    v_num = EGO_ID + 1
    room = (params.length / 2.0) / max(1, params.num_others)
    for i in 1:params.num_others
        type = rand()
        if !params.random
            type = 0.0
        end

        v0 = rand() * params.v_des
        s0 = i * room
        lane0 = LaneTag(rand(1:(4*params.stadium + 1*!params.stadium)), rand(1:params.lanes))
        t0 = 0.0
        ϕ0 = 0.0
        posF = Frenet(roadway[lane0], s0, t0, ϕ0)

        push!(scene, Vehicle(VehicleState(posF, roadway, 0.0), VehicleDef(), v_num))
        if type <= 0.05
            models[v_num] = MPCDriver(params.dt)
            v0 = 0.0
            carcolours[v_num] = MONOKAI["color3"]
        elseif type > 0.05 && type <= 0.5
            models[v_num] = Tim2DDriver(params.dt)
            carcolours[v_num] = MONOKAI["color4"]
        else
            models[v_num] = LatLonSeparableDriver( # produces LatLonAccels
                    ProportionalLaneTracker(), # lateral model
                    IntelligentDriverModel(), # longitudinal model
                    )
            carcolours[v_num] = MONOKAI["color5"]
        end
        AutomotiveDrivingModels.set_desired_speed!(models[v_num], v0)
        v_num += 1
    end

    (scene, models, carcolours)
end

function get_neighbour_features(env::EnvState)
    # ego = env.scene[findfirst(EGO_ID, env.scene)]
    ego_idx = findfirst(EGO_ID, env.scene)
    ego = env.scene[findfirst(EGO_ID, env.scene)]

    road_proj = proj(ego.state.posG, env.roadway)

    left_lane_exists = road_proj.tag.lane < env.params.lanes
    right_lane_exists = road_proj.tag.lane > 1
    fore_M = get_neighbor_fore_along_lane(env.scene, ego_idx, env.roadway, VehicleTargetPointFront(), VehicleTargetPointRear(), VehicleTargetPointFront(), max_distance_fore=env.params.length)
    fore_L = get_neighbor_fore_along_left_lane(env.scene, ego_idx, env.roadway, VehicleTargetPointRear(), VehicleTargetPointRear(), VehicleTargetPointFront(), max_distance_fore=env.params.length)
    fore_R = get_neighbor_fore_along_right_lane(env.scene, ego_idx, env.roadway, VehicleTargetPointRear(), VehicleTargetPointRear(), VehicleTargetPointFront(), max_distance_fore=env.params.length)
    rear_M = get_neighbor_rear_along_lane(env.scene, ego_idx, env.roadway, VehicleTargetPointFront(), VehicleTargetPointFront(), VehicleTargetPointRear(), max_distance_rear=env.params.length)
    rear_L = get_neighbor_rear_along_left_lane(env.scene, ego_idx, env.roadway, VehicleTargetPointFront(), VehicleTargetPointFront(), VehicleTargetPointRear(), max_distance_rear=env.params.length)
    rear_R = get_neighbor_rear_along_right_lane(env.scene, ego_idx, env.roadway, VehicleTargetPointFront(), VehicleTargetPointFront(), VehicleTargetPointRear(), max_distance_rear=env.params.length)

    features = [fore_M.Δs, fore_L.Δs, fore_R.Δs,
                rear_M.Δs, rear_L.Δs, rear_R.Δs]
    features /= env.params.length
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
