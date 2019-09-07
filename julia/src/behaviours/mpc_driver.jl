using Distributions
using Printf

const CAR_LENGTH = 4.0

include("../mpc/utils.jl")
include("../mpc/motion_model.jl")
include("../mpc/mpc_traj_gen.jl")

"""
	MPCDriver <: LaneFollowingDriver

This driver model looks ahead some distance in its own lane, and its
neighbouring lanes, to pick the one with the most free space as the target lane.
The driver then solves for a trajectory to this target in its Frenet frame using
a model predictive control (MPC) optimisation, and executes the first control.

# Constructors
	MPCDriver(timestep::Float64;
	rec::SceneRecord=SceneRecord(1, timestep),
	σ::Float64=1.0,
	num_params::Int64=6,
	lookahead::Float64=50.0,
	v_des::Float64=10.0,
	)

# Fields
- `timestep::Float64` Simulation timestep. Also used as MPC timetep
- `rec::SceneRecord` A record that will hold the resulting simulation results
- `σ::Float64 = 1.0` PDF variance
- `num_params::Int64 = 6` MPC optimisation parameters (fixed)
- `lookahead::Float64 = 50.0` Lane lookahead distance
- `a::Float64` Longitudinal acceleration output
- `δ::Float64` Lateral acceleration output
- `n::Int64 = 26` Number of linearly separater MPC trajectory points
- `interp::Int64 = 1` MPC time-parameterised spline interpolation order
"""
mutable struct MPCDriver <: DriverModel{LatLonAccel}
    rec::SceneRecord
    σ::Float64
	lookahead::Float64

    # Outputs
    a::Float64
	δ::Float64

    # MPC Hyperparameters
    num_params::Int64
    n::Int64
	timestep::Float64
    # time::Float64
	interp::Int64
    weight::Bool

    # Acceleration limits
    amax::Float64
    amin::Float64

    function MPCDriver(
        timestep::Float64;
        rec::SceneRecord=SceneRecord(1, timestep),
        σ::Float64=1.0,
        num_params::Int64=8,
        weight::Bool=false,
		lookahead::Float64=50.0,
        amax::Float64=3.0,
        amin::Float64=9.0
        )
        retval = new()

        retval.rec = rec
        retval.σ = σ
		retval.lookahead = lookahead

        retval.a = NaN
        retval.δ = NaN

        retval.n = 15
		retval.timestep = timestep
        # retval.time = 5.0
		retval.interp = 3
        retval.num_params = Bool(num_params % 2) ? num_params + 1 : num_params
        retval.weight = weight

        retval.amax = amax
        retval.amin = amin

        retval
    end
end
get_name(::MPCDriver) = "MPCDriver"

"""
    set_desired_speed!(model::MPCDriver, v_des::Float64)
MPCDriver does not have or need a desired speed parameter
"""
function AutomotiveDrivingModels.set_desired_speed!(model::MPCDriver, v_des::Float64)
    model
end

"""
	lane_tag_modifier(right::Bool, left::Bool, rΔ::Float64, lΔ::Float64,
							mΔ::Float64)
Select target lane based on available room in right, middle and left lanes
"""
function lane_tag_modifier(right::Bool, left::Bool,
						rΔ_fore::Float64, lΔ_fore::Float64, mΔ_fore::Float64,
						rΔ_rear::Float64, lΔ_rear::Float64, mΔ_rear::Float64)
	if left &&
		lΔ_fore > mΔ_fore + CAR_LENGTH * 2.0 &&
		lΔ_rear > mΔ_rear + CAR_LENGTH * 2.0
		if right
			if lΔ_fore > rΔ_fore + CAR_LENGTH * 2.0 &&
				lΔ_rear > rΔ_rear + CAR_LENGTH * 2.0
				return 1, min(lΔ_fore, lΔ_rear)
			end
		end
	end

	if right &&
		rΔ_fore > mΔ_fore + CAR_LENGTH * 2.0 &&
		rΔ_rear > mΔ_rear + CAR_LENGTH * 2.0
		if left
			if rΔ_fore > lΔ_fore + CAR_LENGTH * 2.0 &&
				rΔ_rear > lΔ_rear + CAR_LENGTH * 2.0
				return -1, min(rΔ_fore, rΔ_rear)
			end
		end
	end

	return 0, min(mΔ_fore, mΔ_rear)
end

"""
	observe!(driver::MPCDriver, scene::Scene, roadway::Roadway, egoid::Int)
1. Get lane information
2. Pick target state in the egovehicle's Frenet coordinate system
3. Solve MPC optimisation to get (and execute) first control input
"""
function AutomotiveDrivingModels.observe!(
				driver::MPCDriver, scene::Scene, roadway::Roadway, egoid::Int)
    update!(driver.rec, scene)

    self_idx = findfirst(egoid, scene)
    ego_state = scene[self_idx].state
	ego_lane = roadway[ego_state.posF.roadind.tag]

	# Step 1
	left_exists = n_lanes_left(ego_lane, roadway) > 0
    right_exists = n_lanes_right(ego_lane, roadway) > 0
	fore_M = get_neighbor_fore_along_lane(scene, self_idx, roadway,
                VehicleTargetPointRear(), VehicleTargetPointRear(),
                VehicleTargetPointRear(), max_distance_fore=driver.lookahead)
    fore_L = get_neighbor_fore_along_left_lane(scene, self_idx, roadway,
                VehicleTargetPointRear(), VehicleTargetPointRear(),
                VehicleTargetPointRear(), max_distance_fore=driver.lookahead)
    fore_R = get_neighbor_fore_along_right_lane(scene, self_idx, roadway,
                VehicleTargetPointRear(), VehicleTargetPointRear(),
                VehicleTargetPointRear(), max_distance_fore=driver.lookahead)
    rear_M = get_neighbor_rear_along_lane(scene, self_idx, roadway,
                VehicleTargetPointFront(), VehicleTargetPointFront(),
                VehicleTargetPointFront(), max_distance_rear=driver.lookahead)
    rear_L = get_neighbor_rear_along_left_lane(scene, self_idx, roadway,
                VehicleTargetPointFront(), VehicleTargetPointFront(),
                VehicleTargetPointFront(), max_distance_rear=driver.lookahead)
    rear_R = get_neighbor_rear_along_right_lane(scene, self_idx, roadway,
                VehicleTargetPointFront(), VehicleTargetPointFront(),
                VehicleTargetPointFront(), max_distance_rear=driver.lookahead)

	# Step 2
	lane_choice, headway = lane_tag_modifier(right_exists, left_exists,
											fore_R.Δs, fore_L.Δs, fore_M.Δs,
											rear_R.Δs, rear_L.Δs, rear_M.Δs)
	headway = min(headway/2.0, driver.lookahead)
	target_lane = roadway[LaneTag(ego_lane.tag.segment, ego_lane.tag.lane + lane_choice)]
	ego_target = Frenet(ego_state.posG, target_lane, roadway) # egostate projected onto target lane
	target_roadind = move_along(ego_target.roadind, roadway, headway) # RoadIndex after moving along target lane
	target_pos = Frenet(target_roadind, roadway) # Frenet position on target lane after moving

	target = MPCState()
	target.x = headway/2.0
	target.y = lane_choice * target_lane.width + ego_state.posF.t
	target.θ = 0.0
	target.v = 0.0
	target.β = 0.0

	# Step 3
	self = MPCState(x=0.0, y=0.0, θ=ego_state.posF.ϕ, v=ego_state.v, β=0.0)
    params = zeros(driver.num_params)
    hyperparams = [driver.n, driver.timestep, driver.interp]
    params, a1, δ1, s_fin, _ = optimise_trajectory(target, params, hyperparams,
                                            initial=self, weight=driver.weight)

	# @printf("(%2.2f, %2.2f, %2.2f, %2.2f, %2.2f)\n", ego_state.v, target.v, s_fin.v, a1, δ1)

	driver.a = a1
	driver.δ = δ1
end

function AutomotiveDrivingModels.observe!(
        driver::MPCDriver, scene::Scene, roadway::Roadway,
        egoid::Int, start::Frenet, v::Float64, goal::Frenet,
        mpc_cf::Float64, mpc_cm::Int)
    # Get trajectory to target point
    traj, a1, δ1 = get_mpc_trajectory(driver, scene, roadway, egoid,
                                start, v, goal)

    # Extract mpc_cf fraction of trajectory for collision checking
    elems = max(Int(ceil(mpc_cf * length(traj))), 1)
    traj_to_check = traj[1:elems]

    # Check for collision based on mpc_cm
    ego = scene[findfirst(egoid, scene)]
    ego_s = ego.state.posF.s
    ego_t = ego.state.posF.t
    ego_ϕ = ego.state.posF.ϕ
    ego_lane = get_lane(roadway, ego.state)
    fake_ego = nothing

    if mpc_cm == 1 # static
        for (i, veh) in enumerate(scene)
            if veh.id ≠ egoid && veh.id ≤ 100
                # check first state for collision
                dist = collision_check(ego, veh)
                if dist ≤ 0
                    driver.a = max(-driver.amin, -v/driver.timestep)
                    driver.δ = 0.0
                    return
                end

                for p in 2:length(traj_to_check)
                    pt = traj_to_check[p]
                    ego_pos = Frenet(ego_lane, ego_s + pt.x, ego_t + pt.y, pt.θ)
                    fake_ego = Vehicle(VehicleState(ego_pos, roadway, 0.0),
                                                            VehicleDef(), 999)

                    dist = collision_check(fake_ego, veh)
                    if dist ≤ 0
                        driver.a = max(-driver.amin, -v/driver.timestep)
                        driver.δ = 0.0
                        return
                    end
                end
            end
        end
    else # constant velocity
        for (i, veh) in enumerate(scene)
            if veh.id ≠ egoid && veh.id ≤ 100
                # check first state for collision
                dist = collision_check(ego, veh)
                if dist ≤ 0
                    driver.a = max(-driver.amin, -v/driver.timestep)
                    driver.δ = 0.0
                    return
                end

                for p in 2:length(traj_to_check)
                    ΔT = driver.timestep * p

                    pt = traj_to_check[p]
                    ego_pos = Frenet(ego_lane, ego_s + pt.x, ego_t + pt.y, pt.θ)
                    fake_ego = Vehicle(VehicleState(ego_pos, roadway, 0.0),
                                                            VehicleDef(), 999)

                    veh′ = propagate(veh, LatLonAccel(0.0, 0.0), roadway, ΔT)
                    fake_veh = Vehicle(veh′, VehicleDef(), 998)

                    dist = collision_check(fake_ego, fake_veh)
                    if dist ≤ 0
                        driver.a = max(-driver.amin, -v/driver.timestep)
                        driver.δ = 0.0
                        return
                    end
                end
            end
        end
    end
    driver.a = min(driver.amax, a1)
    driver.δ = δ1
end

function get_mpc_trajectory(driver::MPCDriver, scene::Scene, roadway::Roadway,
							egoid::Int, start::Frenet, v::Float64, goal::Frenet)
	target = MPCState()
	target.x = goal.s - start.s
	target.y = goal.t - start.t
	target.θ = 0.0
	target.v = 0.0
	target.β = 0.0

	# Step 3
	self = MPCState(x=0.0, y=0.0, θ=start.ϕ, v=v, β=0.0)
	params = zeros(driver.num_params)
	hyperparams = [driver.n, driver.timestep, driver.interp]
	_, a1, δ1, _, states = optimise_trajectory(target, params, hyperparams,
                                            initial=self, weight=driver.weight)

	(states, a1, δ1)
end

function set_hyperparams!(model::MPCDriver, hyperparams::Vector{Float64})
    n, timestep, interp = hyperparams
    model.n = Int64(n)
    model.timestep = timestep
    model.interp = Int64(interp)
    model
end
function set_hyperparams!(model::MPCDriver, n::Int64, timestep::Float64,
                            interp::Int64)
    model.n = n
    model.timestep = timestep
    model.interp = interp
    model
end
function set_σ!(model::MPCDriver, σ::Float64)
    model.σ = σ
    model
end

function Base.rand(model::MPCDriver)
    if isnan(model.a) || isnan(model.δ)
        LatLonAccel(0.0, 0.0)
    else
        LatLonAccel(model.δ, model.a)
    end
end
function Distributions.pdf(model::MPCDriver, a::LatLonAccel)
    if isnan(model.a) || isnan(model.δ)
        Inf
    else
        pdf(Normal(model.a, model.σ), a.a_lon) * pdf(Normal(model.δ, model.σ), a.a_lat)
    end
end
function Distributions.logpdf(model::MPCDriver, a::LatLonAccel)
    if isnan(model.a) || isnan(model.δ)
        Inf
    else
        logpdf(Normal(model.a, model.σ), a.a_lon) * logpdf(Normal(model.δ, model.σ), a.a_lat)
    end
end
