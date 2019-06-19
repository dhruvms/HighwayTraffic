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
    length = get(params, "length", 200.0)
    lanes = get(params, "lanes", 2)
    cars = get(params, "cars", 1)
    v_des = get(params, "v_des", 10.0)
    dt = get(params, "dt", 0.2)
    o_dim = get(params, "o_dim", 7)

    a_cost = get(params, "a_cost", 0.1)
    δ_cost = get(params, "d_cost", 0.01)
    v_cost = get(params, "v_cost", 0.2)
    ϕ_cost = get(params, "phi_cost", 0.35)
    t_cost = get(params, "t_cost", 0.34)

    num_others = get(params, "num_others", 4)
    random = get(params, "random_others", true)

    EnvParams(length, lanes, cars, v_des, dt, o_dim,
                a_cost, δ_cost, v_cost, ϕ_cost, t_cost,
                num_others, random)
end

function get_initial_egostate(params::EnvParams, roadway::Roadway{Float64})
    v0 = rand() * params.v_des
    s0 = rand() * (params.length / 4.0)
    t0 = (2 * DEFAULT_LANE_WIDTH * rand()) - (DEFAULT_LANE_WIDTH/2.0)
    ϕ0 = (2 * rand() - 1) * 0.6 # max steering angle
    ego = Entity(AgentState(roadway, v=v0, s=s0, t=t0, ϕ=ϕ0), EgoVehicle(), EGO_ID)
    return Frame([ego])
end

function populate_others(params::EnvParams, roadway::Roadway{Float64})
    scene = Scene()
    carcolours = Dict{Int, Colorant}()
    models = Dict{Int, DriverModel}()

    v_num = EGO_ID + 1
    for i in 1:params.num_others
        type = rand()
        if !params.random
            type = 0.0
        end

        v0 = rand() * params.v_des
        s0 = rand() * (params.length / 4.0)
        lane0 = LaneTag(1, rand(1:params.lanes))
        posF = Frenet(roadway[lane0], s0, t=0.0, ϕ=0.0)

        push!(scene, Vehicle(VehicleState(posF, roadway, v0), VehicleDef(), v_num))
        if type <= 0.05
            models[v_num] = MPCDriver(timestep)
            v = 0.0
            carcolours[v_num] = MONOKAI["color3"]
        elseif type > 0.05 && type <= 0.5
            models[v_num] = Tim2DDriver(timestep)
            carcolours[v_num] = MONOKAI["color4"]
        else
            models[v_num] = LatLonSeparableDriver( # produces LatLonAccels
                    ProportionalLaneTracker(), # lateral model
                    IntelligentDriverModel(), # longitudinal model
                    )
            carcolours[v_num] = MONOKAI["color5"]
        end
        set_desired_speed!(models[v_num], v)
        v_num += 1
    end

    (scene, models, carcolours)
end

function get_neighbour_features(env::EnvState)
    veh = env.scene[findfirst(EGO_ID, env.scene)]
    road_proj = proj(veh.state.state.posG, env.roadway)

    left_lane_exists = road_proj.tag.lane < env.params.lanes
    right_lane_exists = road_proj.tag.lane > 1
    fore_M = get_neighbor_fore_along_lane(env.scene, EGO_ID, env.roadway, VehicleTargetPointFront(), VehicleTargetPointRear(), VehicleTargetPointFront(), max_distance_fore=env.params.length)
    fore_L = get_neighbor_fore_along_left_lane(env.scene, EGO_ID, env.roadway, VehicleTargetPointRear(), VehicleTargetPointRear(), VehicleTargetPointFront(), max_distance_fore=env.params.length)
    fore_R = get_neighbor_fore_along_right_lane(env.scene, EGO_ID, env.roadway, VehicleTargetPointRear(), VehicleTargetPointRear(), VehicleTargetPointFront(), max_distance_fore=env.params.length)
    rear_M = get_neighbor_rear_along_lane(env.scene, EGO_ID, env.roadway, VehicleTargetPointFront(), VehicleTargetPointFront(), VehicleTargetPointRear(), max_distance_rear=env.params.length)
    rear_L = get_neighbor_rear_along_left_lane(env.scene, EGO_ID, env.roadway, VehicleTargetPointFront(), VehicleTargetPointFront(), VehicleTargetPointRear(), max_distance_rear=env.params.length)
    rear_R = get_neighbor_rear_along_right_lane(env.scene, EGO_ID, env.roadway, VehicleTargetPointFront(), VehicleTargetPointFront(), VehicleTargetPointRear(), max_distance_rear=env.params.length)

    features = [fore_M.Δs, fore_L.Δs, fore_R.Δs,
                rear_M.Δs, rear_L.Δs, rear_R.Δs]
    features /= env.params.length
    features
end

function is_crash(scene::Scene)
    ego = scene[findfirst(EGO_ID, scene)]

    if ego.state.v ≈ 0
        return false
    end
    for veh in scene
        if veh.id != EGO_ID
            if is_colliding(ego, veh)
                return true
            end
        end
    end
    return false
end
