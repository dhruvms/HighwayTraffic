#!/bin/bash

gen_results()
{
        EXP=$(echo $1 | sed -n "s/^.*curriculum\/\(\S*\).*$/\1/p")
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

        simcars=$2
        gap=$3
        echo "simcars: " ${simcars} ", gap: " ${gap}
        echo
        # python enjoy.py --env-name HighwayTraffic-v1 --algo ppo \
        #         --lr 2.5e-4 --num-processes 1 --cars ${CARS} --gap 1.1 \
        #         --base-port 9999 --length 1000.0 --lanes ${LANES} --change \
        #         --beta-dist --occupancy --seed ${SEED} --hri \
        #         --eval-episodes 200 --write-data --max-steps 200 \
        #         --eval-mode mixed --eval-folder mixed \
        #         --testcars 60 --stopgo
        # kill -9 $(pgrep -f "port 9999")

        # python enjoy.py --env-name HighwayTraffic-v1 --algo ppo \
        #         --lr 2.5e-4 --num-processes 1 --cars ${CARS} --gap 1.1 \
        #         --base-port 9999 --length 1000.0 --lanes ${LANES} --change \
        #         --beta-dist --occupancy --seed ${SEED} --hri \
        #         --eval-episodes 200 --write-data --max-steps 200 \
        #         --eval-mode cooperative --eval-folder cooperative \
        #         --testcars 60 --stopgo
        # kill -9 $(pgrep -f "port 9999")

        # python enjoy.py --env-name HighwayTraffic-v1 --algo ppo \
        #         --lr 2.5e-4 --num-processes 1 --cars ${CARS} --gap 1.1 \
        #         --base-port 9999 --length 1000.0 --lanes ${LANES} --change \
        #         --beta-dist --occupancy --seed ${SEED} --hri \
        #         --eval-episodes 200 --write-data --max-steps 200 \
        #         --eval-mode aggressive --eval-folder aggressive \
        #         --testcars 60 --stopgo
        # kill -9 $(pgrep -f "port 9999")

        # python results.py --cars ${CARS} --length 1000.0 --lanes ${LANES} \
        #         --change --beta-dist --lr 2.5e-4 --seed ${SEED}

        python enjoy.py --env-name HighwayTraffic-v1 --algo ppo \
                --lr 2.5e-4 --num-processes 1 --cars ${CARS} \
                --base-port 9999 --length 1000.0 --lanes ${LANES} --change \
                --beta-dist --occupancy --seed ${SEED} --hri \
                --eval-episodes 200 --write-data --max-steps 200 \
                --eval-mode mixed --eval-folder mixed \
                --testcars ${simcars} --gap ${gap} #--stopgo
        kill -9 $(pgrep -f "port ${port}")
}

declare -a cars=(15 30 45 60 75 90)
declare -a gaps=(1.1 1.5 2.0 2.5 3.0)

ROOT=$1
for EXPDIR in $(find ${ROOT} -mindepth 1 -maxdepth 1 -type d);
do
        for c in "${cars[@]}"
        do
            for g in "${gaps[@]}"
                do
                        gen_results "$EXPDIR" "$c" "$g"
                done
        done
        # gen_results "$EXPDIR"
done
