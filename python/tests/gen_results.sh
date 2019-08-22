#!/bin/bash

gen_results()
{
        EXP=$(echo $1 | sed -n "s/^.*present\/\(\S*\).*$/\1/p")
        if [[ ${EXP} == "old" ]]; then
                echo "Skip: " ${EXP}
                echo
                return
        fi
        LANES=$(echo ${EXP} | sed -n "s/^.*Length_\(\S*\)-Lanes.*$/\1/p")
        CARS=$(echo ${EXP} | sed -n "s/^.*Lanes_\(\S*\)-Cars.*$/\1/p")
        SEED=$(echo ${EXP} | sed -n "s/^.*Seed-\(\S*\)_VCost.*$/\1/p")
        echo ${EXP}
        echo "Cars: " ${CARS} ", Lanes: " ${LANES} ", Seed: " ${SEED}
        echo

        python enjoy.py --env-name LaneFollow-v1 --algo ppo --num-processes 1 \
                --eval-episodes 200 --base-port 9999 --cars ${CARS} \
                --length 1000.0 --lanes ${LANES} --change --beta-dist \
                --occupancy --max-steps 200 --lr 2.5e-4 --seed ${SEED} \
                --write-data --hri \ # include --video to save videos
                --eval-mode mixed --eval-folder mixed
        kill -9 $(pgrep -f "port 9999")

        python enjoy.py --env-name LaneFollow-v1 --algo ppo --num-processes 1 \
                --eval-episodes 200 --base-port 9999 --cars ${CARS} \
                --length 1000.0 --lanes ${LANES} --change --beta-dist \
                --occupancy --max-steps 200 --lr 2.5e-4 --seed ${SEED} \
                --write-data --hri \ # include --video to save videos
                --eval-mode cooperative --eval-folder cooperative
        kill -9 $(pgrep -f "port 9999")

        python enjoy.py --env-name LaneFollow-v1 --algo ppo --num-processes 1 \
                --eval-episodes 200 --base-port 9999 --cars ${CARS} \
                --length 1000.0 --lanes ${LANES} --change --beta-dist \
                --occupancy --max-steps 200 --lr 2.5e-4 --seed ${SEED} \
                --write-data --hri \ # include --video to save videos
                --eval-mode aggressive --eval-folder aggressive
        kill -9 $(pgrep -f "port 9999")

        python results.py --cars ${CARS} --length 1000.0 --lanes ${LANES} \
                --change --beta-dist --lr 2.5e-4 --seed ${SEED}
}

ROOT=$1
for EXPDIR in $(find ${ROOT} -mindepth 1 -maxdepth 1 -type d);
do
        gen_results $EXPDIR
done
