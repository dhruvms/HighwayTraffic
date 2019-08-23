# Deep Reinforcement Learning for Driving in Dense Traffic
An actor-critic based deep reinforcement learning solution for autonomous driving on roads with dense traffic.

D. M. Saxena, A. N. Sarveani, S. Bae, K. Fujimura, and M. Likhachev, _Deep Reinforcement Learning for Driving in Dense Traffic_, in preparation.

## Prerequisites
- Julia 1.1.1
- Python 3.5+
- PyTorch 1.1.0
- TensorFlow CPU for TensorBoard logging

## Dependencies

### Julia
- [Automotive Driving Models](https://github.com/sisl/AutomotiveDrivingModels.jl/)
- [AutoViz](https://github.com/sisl/AutoViz.jl)
- Reel, `] add Reel`
- Distributions, `] add Distributions`
- Parameters, `] add Parameters`
- Dierckx, `] add Dierckx`
- ZMQ, `] add ZMQ`
- JSON, `] add JSON`
- [RLInterface](https://github.com/JuliaPOMDP/RLInterface.jl)
- ArgParse, `] add ArgParse`
- Other Julia modules: Print, DelimitedFiles, Statistics, Random, LinearAlgebra

`src/behaviors/intelligent_driver_model.jl` needs to be edited in AutomotiveDrivingModels.jl. In order to do this, first checkout the package in development mode by executing `] dev AutomotiveDrivingModels`. Then make the following edits,

```
@@ -33,6 +33,8 @@ around the non-errorable IDM output.
     a_max::Float64 = 3.0 # maximum acceleration ability [m/s²]
     d_cmf::Float64 = 2.0 # comfortable deceleration [m/s²] (positive)
     d_max::Float64 = 9.0 # maximum deceleration [m/s²] (positive)
+
+    ΔT::Float64 = 0.2 # simulation timestep
 end
 get_name(::IntelligentDriverModel) = "IDM"
 function set_desired_speed!(model::IntelligentDriverModel, v_des::Float64)
@@ -64,6 +66,10 @@ function track_longitudinal!(model::IntelligentDriverModel, v_ego::Float64, v_ot

     model.a = clamp(model.a, -model.d_max, model.a_max)

+    if v_ego + model.ΔT * model.a < 0
+        model.a = max(-model.d_max, -v_ego/model.ΔT)
+    end
+
     return model
 end
 function Base.rand(model::IntelligentDriverModel)
```

### Python
- [OpenAI Gym](https://github.com/openai/gym)
- [Baselines](https://github.com/openai/baselines)
- [PPO](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) by @ikostrikov
    - the required files are included in this repository [here](./python/external/pytorch_baselines/).

## Project structure
- `python`: contains all Python code
    - `algs/`: training scripts for PPO and DDPG (out-of-date)
    - `envs/`: Python wrappers around Julia environments
    - `external/`: files from the [PyTorch baselines repository](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) by @ikostrikov
    - `models/`: out-of-date Actor-Critic implementation for DDPG
    - `structs/`: replay memory for DDPG (out-of-date)
    - `tests/`: scripts to generate results (numbers and videos both), along with screen files for experiments
    - `utils/`: some DDPG utilities (out-of-date) and TensorBoard logger
- `julia`: contains all Julia code
    - `scripts/zmq_server.jl`: launches an instance of the simulator in Julia ready for TCP/IP communication
    - `test/mpc.jl`: includes a test script for the MPCDriver
    - `src/`: all code that interacts with the Julia simulator and Python pipeline
        - `agent/`: egovehicle model with bicycle kinematics
        - `behaviours/`: [BafflingDriver](https://github.com/honda-research-institute/NNMPC.jl/) (relevant files have been copied) and MPCDriver models
        - `envs/`: Julia environment structure and helper functions
        - `mpc/`: backbone and helpers for the MPCDriver model

## Example

A training session can be launched by following the example below. The usage of different parameters can be understood in [this file](./python/external/pytorch_baselines/a2c_ppo_acktr/arguments.py)
```
python train_baseline.py \
                            --env-name LaneFollow-v1 \
                            --algo ppo \
                            --use-gae \
                            --lr 2.5e-4 \
                            --use-linear-lr-decay \
                            --clip-param 0.1 \
                            --value-loss-coef 0.5 \
                            --num-processes 8 \
                            --num-steps 128 \
                            --num-mini-batch 4 \
                            --log \
                            --log-interval 10 \
                            --entropy-coef 0.01 \
                            --cars 60 \
                            --base-port 9300 \
                            --length 1000.0 \
                            --lanes 2 \
                            --change \
                            --beta-dist \
                            --occupancy \
                            --gamma 0.995 \
                            --seed 794 \
                            --hri \
                            --curriculum
```

## Author

Developed by [Dhruv Mauria Saxena](mailto:dhruvsaxena@cmu.edu) as part of his internship at Honda Research
Institute, USA.

## Acknowledgements

- Supervisor: Alireza Nakhaei Sarvedani
