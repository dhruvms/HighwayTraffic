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
