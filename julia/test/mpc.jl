using AutomotiveDrivingModels
using AutoViz
using Distributions
using Interact

include("../src/behaviours/mpc_driver.jl")

# roadway parameters
lanes = 4
road_length = 200.0
width = 10.0
radius = 20.0
roadway = gen_stadium_roadway(lanes, length=road_length, width=width, radius=radius)

# simulation parameters
num_vehs = 50
timestep = 0.2
max_v = 15.0

# populate scene
scene = Scene()
carcolors = Dict{Int,Colorant}()
models = Dict{Int, DriverModel}()
offset = (road_length/(num_vehs/lanes))
v_num = 1
for i in 1:(num_vehs/lanes)
    x_offset = (i-1) * offset
    for j in 1:lanes
        type = rand()
        side = rand()

        x = x_offset
        y = 0.0
        th = 0.0
        if side > 0.5
            y += width + 2*radius + (j-1) * DEFAULT_LANE_WIDTH
            th = π
        else
            y += (1-j) * DEFAULT_LANE_WIDTH
        end
        v = max(1.0, rand() * max_v) * 2.0

        push!(scene, Vehicle(VehicleState(VecSE2(x, y, th), roadway, 0.0), VehicleDef(), v_num))
        if type <= 0.05
            models[v_num] = MPCDriver(timestep)
            v = 0.0
            carcolors[v_num] =  try
                                    MONOKAI["color3"]
                                catch
                                    MONOKAY["color3"]
                                end
        elseif type > 0.05 && type <= 0.5
            models[v_num] = Tim2DDriver(timestep,
                    mlane = MOBIL(timestep)
                    )
            carcolors[v_num] =  try
                                    MONOKAI["color4"]
                                catch
                                    MONOKAY["color4"]
                                end
        else
            models[v_num] = LatLonSeparableDriver( # produces LatLonAccels
                    ProportionalLaneTracker(), # lateral model
                    IntelligentDriverModel(ΔT = timestep), # longitudinal model
                    )
            carcolors[v_num] =  try
                                    MONOKAI["color5"]
                                catch
                                    MONOKAY["color5"]
                                end
        end
        AutomotiveDrivingModels.set_desired_speed!(models[v_num], v)
        global v_num += 1
    end
end
cam = FitToContentCamera(0.01)

# simulate
nticks = 200
rec = SceneRecord(nticks+1, timestep)
simulate!(rec, scene, roadway, models, nticks)
# render(rec[0], roadway, cam=cam, car_colors=carcolors)

# # render
# @manipulate for frame_index in 1:nframes(rec)
#     render(rec[frame_index-nframes(rec)], roadway, cam=cam, car_colors=carcolors)
# end

using Blink
using Interact
w = Window()
ui = @manipulate for frame_index in 1:nframes(rec)
    render(rec[frame_index-nframes(rec)], roadway, cam=cam, car_colors=carcolors)
end
body!(w, ui)
