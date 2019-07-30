using AutomotiveDrivingModels
using LinearAlgebra

include("baffling_lateral_tracker.jl")
include("baffling_longitudinal_tracker.jl")
include("baffling_lane_changer.jl")

"""
	BafflingDriver
Driver that randomly changes lanes and speeds.

# Constructors
	BafflingDriver(timestep::Float64;mlon::LaneFollowingDriver=IntelligentDriverModel(), mlat::LateralDriverModel=ProportionalLaneTracker(), mlane::LaneChangeModel=RandLaneChanger(timestep),rec::SceneRecord = SceneRecord(1, timestep))

# Fields
- `rec::SceneRecord` A record that will hold the resulting simulation results
- `mlon::LaneFollowingDriver = IntelligentDriverModel()` Longitudinal driving model
- `mlat::LateralDriverModel = ProportionalLaneTracker()` Lateral driving model
- `mlane::LaneChangeModel =RandLaneChanger` Lane change model (randomly)
"""
mutable struct BafflingDriver <: DriverModel{LatLonAccel}
    rec::SceneRecord
    mlon::LaneFollowingDriver
    mlat::LateralDriverModel
    mlane::LaneChangeModel
    dt::Float64
    η_coop::Float64 # cooperativeness parameter. η_coop==1 : yield 100%, η_coop==0 : go 100% >> Bernoulli parameter
    η_percept::Float64
    r::Float64 # randomness
    allowLaneChange::Bool # allow lane changing

    function BafflingDriver(
        timestep::Float64;
        η_coop::Float64 = 0.5, # cooperativeness [0,1]
        η_percept::Float64 = 0.3, # perception range [meter]
        r::Float64 = 0.04,
        mlon::LaneFollowingDriver=BafflingLongitudinalTracker(),
        mlat::LateralDriverModel=BafflingLateralTracker(),
        mlane::LaneChangeModel=BafflingLaneChanger(timestep,threshold_lane_change_rand = r),
        rec::SceneRecord = SceneRecord(1, timestep),
        allowLaneChange = false
        )

        retval = new()

        retval.rec = rec
        retval.mlon = mlon
        retval.mlat = mlat
        retval.mlane = mlane
        retval.dt = timestep
        retval.η_coop = η_coop
        retval.η_percept = η_percept
        retval.r = r
        retval.allowLaneChange = allowLaneChange

        retval
    end
end
get_name(::BafflingDriver) = "BafflingDriver"
function AutomotiveDrivingModels.set_desired_speed!(model::BafflingDriver, v_des::Float64)
    set_desired_speed!(model.mlon, v_des)
    set_desired_speed!(model.mlane, v_des)
    model
end

function AutomotiveDrivingModels.propagate(veh::Entity{VehicleState, VehicleDef, Int}, action::LatLonAccel, roadway::Roadway, ΔT::Float64; Δϕ_max::Float64 = 0.4)
    a_lat = action.a_lat
    a_lon = action.a_lon

    v = veh.state.v
    ϕ = veh.state.posF.ϕ
    ds = v*cos(ϕ)
    t = veh.state.posF.t
    dt = v*sin(ϕ)

    ΔT² = ΔT*ΔT
    Δs = ds*ΔT + 0.5*a_lon*ΔT²
    Δt = dt*ΔT + 0.5*a_lat*ΔT²

    ds₂ = ds + a_lon*ΔT
    dt₂ = dt + a_lat*ΔT
    speed₂ = sqrt(dt₂*dt₂ + ds₂*ds₂)
    v₂ = sqrt(dt₂*dt₂ + ds₂*ds₂) # v is the magnitude of the velocity vector
    # ϕ₂ = sign(atan(dt₂, ds₂))*min(abs(atan(dt₂, ds₂)),Δϕ_max) # project to the maximum steering rate
    # ϕ₂ = atan(dt₂, ds₂)
    ϕ₂ = ds₂ <= 0.4 ? 0 : atan(dt₂, ds₂)

    roadind = move_along(veh.state.posF.roadind, roadway, Δs)
    footpoint = roadway[roadind]
    posG = VecE2{Float64}(footpoint.pos.x,footpoint.pos.y) + polar(t + Δt, footpoint.pos.θ + π/2)

    posG = VecSE2{Float64}(posG.x, posG.y, footpoint.pos.θ + ϕ₂)

    state′ = VehicleState(posG, roadway, v₂)
    if abs(state′.posF.ϕ - veh.state.posF.ϕ) > Δϕ_max
        state = veh.state
    else
        state = state′
    end
    return state
end

function AutomotiveDrivingModels.observe!(driver::BafflingDriver, scene::Scene, roadway::Roadway, egoid::Int)

    update!(driver.rec, scene)

    if driver.allowLaneChange
        observe!(driver.mlane, scene, roadway, egoid) # receive action from the lane change controller
        lane_change_action = rand(driver.mlane)
    else
        lane_change_action = LaneChangeChoice(DIR_MIDDLE)
    end

    vehicle_index = findfirst(egoid, scene)
    laneoffset = get_lane_offset(lane_change_action, driver.rec, roadway, vehicle_index)
    lateral_speed = convert(Float64, get(VELFT, driver.rec, roadway, vehicle_index))

    if lane_change_action.dir == DIR_MIDDLE
        # fore = get_neighbor_fore_along_lane(scene, vehicle_index, roadway, VehicleTargetPointFront(), VehicleTargetPointRear(), VehicleTargetPointFront())
        fore = get_neighbor_fore_along_lane(scene, vehicle_index, roadway, η_coop = driver.η_coop, η_percept = driver.η_percept)
    elseif lane_change_action.dir == DIR_LEFT
        fore = get_neighbor_fore_along_left_lane(scene, vehicle_index, roadway, VehicleTargetPointFront(), VehicleTargetPointRear(), VehicleTargetPointFront())
    else
        @assert(lane_change_action.dir == DIR_RIGHT)
        fore = get_neighbor_fore_along_right_lane(scene, vehicle_index, roadway, VehicleTargetPointFront(), VehicleTargetPointRear(), VehicleTargetPointFront())
    end

    AutomotiveDrivingModels.track_lateral!(driver.mlat, laneoffset, lateral_speed) # receive acceleration from the lateral controller
    track_longitudinal!(driver.mlon, scene, roadway, vehicle_index, fore) # receive acceleration from the longitudinal controller

    # println("veh ", egoid, "headway: ", fore.Δs, ", actions: ", rand(driver))
    driver
end

# const HEAD_RIGHT = 1
# const HEAD_LEFT = 2
# const HEAD_UP = 3
# const HEAD_DOWN = 4

function AutomotiveDrivingModels.get_neighbor_fore_along_lane(scene::Scene, vehicle_index::Int, roadway::Roadway;
                    η_coop::Float64 = 0.5, η_percept::Float64 = 0.3, max_distance_fore::Float64 = 50.0, ignore_dist::Float64 = 1.0)
    ego = scene[vehicle_index]
    x = ego.state.posG.x
    y = ego.state.posG.y
    θ = ego.state.posG.θ
    current_lane = scene[vehicle_index].state.posF.roadind.tag.lane

    # list up vehicles of which head/rear are in the same lane
    inds_vehs_watching = []
    for (i,veh) in enumerate(scene)
        i == vehicle_index && continue
        isFront = dot([veh.state.posG.x-x, veh.state.posG.y-y], [cos(θ),sin(θ)]) > 0

        if isFront && distance_center(ego, veh) <= max_distance_fore
            posG = veh.state.posG
            posF = veh.state.posF
            l = veh.def.length
            w = veh.def.width
            isCenterIn = veh.state.posF.roadind.tag.lane == current_lane
            isAdjacent = distance_perpendicular(ego,veh) < DEFAULT_LANE_WIDTH
            isBodyIn = abs(posF.t) +
                       sqrt((l/2)^2+(w/2)^2)*abs(cos(π/2-(posF.ϕ+atan(w/l)))) >
                       DEFAULT_LANE_WIDTH/2 - η_percept
            if isCenterIn
                append!(inds_vehs_watching, i)
            elseif isAdjacent && isBodyIn && distance_center(ego, veh) > ignore_dist
                isCollisionRisk = abs(posF.t) +
                           sqrt((l/2)^2+(w/2)^2)*abs(cos(π/2-(posF.ϕ+atan(w/l)))) >
                           DEFAULT_LANE_WIDTH - (abs(ego.state.posF.t)+ego.def.width/2)
                if isCollisionRisk
                    append!(inds_vehs_watching, i)
                else
                    # generate yield variable ∈ [0, 1]
                    doYield = rand() <= η_coop
                    doYield && append!(inds_vehs_watching, i)
                end
            end
        end
    end

    best_ind = nothing
    best_dist = max_distance_fore
    for i in inds_vehs_watching
        dis = distance_body(ego, scene[i])
        if dis < best_dist
            best_ind = i
            best_dist = dis
        end
    end
    return NeighborLongitudinalResult(best_ind, best_dist)
end

function distance_perpendicular(ego::Entity{VehicleState, D, Int}, veh::Entity{VehicleState, D, Int}) where {D}
    x = ego.state.posG.x
    y = ego.state.posG.y

    posG = veh.state.posG
    posF = veh.state.posF
    m = sqrt(posG.x^2+posG.y^2)
    n = sqrt(x^2+y^2)
    inner_prod = dot([posG.x,posG.y],[x,y])
    θ_rel = acos(min(inner_prod/(m*n),1))
    dist_perpend = m * sin(θ_rel)
    dist_perpend < 0 && @error "negative distance"
    return dist_perpend
end

function distance_center(ego::Entity{VehicleState, D, Int}, other::Entity{VehicleState, D, Int}) where {D}
    r = ego.def.width/2
    x = ego.state.posG.x
    y = ego.state.posG.y
    xᵢ = other.state.posG.x
    yᵢ = other.state.posG.y
    dis = sqrt(
              ( (x) - (xᵢ) )^2
             +( (y) - (yᵢ) )^2
             ) - 2*r
    return max(dis, 0.0)
end

function distance_body(ego::Entity{VehicleState, D, Int}, other::Entity{VehicleState, D, Int}) where {D}
    r = ego.def.width/2
    h = ego.def.length/2
    x = ego.state.posG.x
    y = ego.state.posG.y
    θ = ego.state.posG.θ
    rᵢ = other.def.width/2
    xᵢ = other.state.posG.x
    yᵢ = other.state.posG.y
    θᵢ = ego.state.posG.θ
    min_dis = Inf
    for i in [-1,0,1]
        for j in [-1,0,1]
            dis = sqrt(
                      ( (x + i*(h-r)*cos(θ)) - (xᵢ+ j*(h-r)*cos(θᵢ)) )^2
                     +( (y + i*(h-r)*sin(θ)) - (yᵢ+ j*(h-r)*sin(θᵢ)) )^2
                     )     - (r + rᵢ)
            min_dis = min(dis, min_dis)
        end
    end
    return max(min_dis, 0)
end

Base.rand(driver::BafflingDriver) = LatLonAccel(rand(driver.mlat), rand(driver.mlon).a)
Distributions.pdf(driver::BafflingDriver, a::LatLonAccel) = pdf(driver.mlat, a.a_lat) * pdf(driver.mlon, a.a_lon)
Distributions.logpdf(driver::BafflingDriver, a::LatLonAccel) = logpdf(driver.mlat, a.a_lat) * logpdf(driver.mlon, a.a_lon)
