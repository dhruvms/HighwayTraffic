using ZMQ
using JSON
using RLInterface: ZMQTransport, sendresp, recvreq, close
using ArgParse

include("../src/envs/TCNData.jl")
using .TCNData

s = ArgParseSettings()
@add_arg_table s begin
    "--port"
        help = "port for the ZMQ communication"
        arg_type = Int
        default = 9393
    "--ip"
        help = "IP address for the ZMQ connection"
        default = "127.0.0.1"
end
parsed_args = parse_args(ARGS, s)

port = parsed_args["port"]
ip = parsed_args["ip"]

function process!(env::TCNEnv, msg::Dict{String, T}) where T
    if "cmd" in keys(msg)
        if msg["cmd"] == "render"
            filename = msg["filename"]
            save_gif(env, filename)
            respmsg = Dict()
        elseif msg["cmd"] == "reset"
            paramdict = msg["params"]
            env, obs = reset(paramdict)
            respmsg = Dict("obs" => obs)
        elseif msg["cmd"] == "step"
            obs, rew, done, info, env = step!(env)
            respmsg = Dict("obs" => obs, "rew" => rew,
                            "done" => done, "info" => info)
        else
            respmsg = Dict("error" => "no known " + msg["cmd"] + " cmd found")
        end
    end
    respmsg, env
end

function run_env_server(ip, port, env::TCNEnv)
    conn = ZMQTransport(ip, port, ZMQ.REP, true)
    @info("running server: ", ip, port)
    while true
        msg = JSON.parse(recvreq(conn))
        # @info("received request: ", msg)
        respmsg, env = process!(env, msg)
        sendresp(conn, respmsg)
    end
    close(conn)
end

env, _ = reset(Dict())
run_env_server(ip, port, env)
