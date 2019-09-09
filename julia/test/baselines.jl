using ArgParse
using Dates

include("../src/envs/HighwayBaselines.jl")
using .HighwayBaselines

s = ArgParseSettings(autofix_names=true)
@add_arg_table s begin
    "--length"
        help = "length of roadway"
        arg_type = Float64
        default = 1000.0
    "--lanes"
        help = "number of lanes in roadway"
        arg_type = Int
        default = 3
    "--cars"
        help = "number of cars on roadway, including egovehicle"
        arg_type = Int
        default = 30
    "--dt"
        help = "timestep"
        arg_type = Float64
        default = 0.2
    "--max-steps"
        help = "max ticks per episode"
        arg_type = Int
        default = 200
    "--stadium"
        help = "stadium roadway"
        arg_type = Bool
        default = false
    "--change"
        help = "change to different lane"
        arg_type = Bool
        default = false
    "--both"
        help = "change and follow"
        arg_type = Bool
        default = false
    "--fov"
        help = "longitudinal field-of-view"
        arg_type = Int
        default = 50
    "--hri"
        help = "HRI specific test case"
        arg_type = Bool
        default = false
    "--curriculum"
        help = "(randomised) curriculum of cars and gaps during training"
        arg_type = Bool
        default = false
    "--gap"
        help = "gap between cars"
        arg_type = Float64
        default = 1.1
    "--ego-model"
        help = "ego model for baseline"
        arg_type = Int64
        default = 1
    "--mpc-s"
        help = "MPC lookahead"
        arg_type = Float64
        default = nothing
    "--mpc-cf"
        help = "MPC collision check fraction"
        arg_type = Float64
        default = nothing
    "--mpc-cm"
        help = "MPC collision check scheme"
        arg_type = Int64
        default = nothing
    "--eval-mode"
        help = "types of other vehicles (mixed/cooperative/aggressive)"
        arg_type = String
        default = "mixed"
    "--video"
        help = "save video"
        arg_type = Bool
        default = false
    "--write-data"
        help = "save data file"
        arg_type = Bool
        default = false


    "--episodes"
        help = "number of test episodes"
        arg_type = Int
        default = 10
    "--save-folder"
        help = "results folder"
        arg_type = String
        default = "../data/"
    "--eval"
        help = "eval mode flag"
        arg_type = Bool
        default = false
end

parsed_args = parse_args(ARGS, s)
results_dir = parsed_args["save_folder"]

lanes = parsed_args["lanes"]
cars = parsed_args["cars"]
gap = parsed_args["gap"]
exp_dir = "$cars-Cars_$lanes-Lanes_$gap-Gap/" * parsed_args["eval_mode"] * "/"
results_dir *= exp_dir
if occursin("../data/", results_dir)
	parsed_args["gap"] += 0.5

if parsed_args["ego_model"] == 1
	results_dir *= "IDM-MOBIL/"
else
	if occursin("data2", results_dir)
		cars = parsed_args["cars"]
		gap = parsed_args["gap"]
		results_dir *= "$cars-Cars_$gap-Gap/"
	else
		s = parsed_args["mpc_s"]
		cf = parsed_args["mpc_cf"]
		cm = parsed_args["mpc_cm"]
		results_dir *= "MPC_$s-$cf-$cm/"
	end
end
results_dir *= Dates.format(Dates.now(), "yyyymmdd-HHMMSS") * "/"
mkpath(results_dir)

for ep = 1:parsed_args["episodes"]
	env = reset(parsed_args)
	terminal = false
	reward = 0.0
	rewards = []
	while !terminal
		r, terminal, env = step!(env)
		reward += r
	end
	push!(rewards, reward)

	save_gif(env, results_dir * "test_$ep.mp4")
end
