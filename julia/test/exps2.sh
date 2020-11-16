#!/bin/bash

exp() {
	# s=$1
	# cf=$2
	# cm=$3
	# d=$4
	# l=$5
	c=$1
	g=$2
	julia baselines.jl \
	--change true \
	--hri true \
	--episodes 200 \
	--ego-model 2 \
	--write-data true \
	--lanes 3 \
	--cars $c \
	--gap $g \
	--eval-mode mixed \
	--mpc-s 3 \
	--mpc-cf 0.25 \
	--mpc-cm 2
}

# declare -a lookahead=(2 3 5)
# declare -a check_fraction=(0.0 0.25 0.5)
# declare -a check_mode=(1 2)
# declare -a drivers=("cooperative" "aggressive" "mixed")
# declare -a lanes=(2 3)
declare -a cars=(15 30 45 60 75 90)
declare -a gaps=(1.1 1.5 2.0 2.5 3.0)

# for s in "${lookahead[@]}"
# do
# 	for cf in "${check_fraction[@]}"
# 	do
# 		for cm in "${check_mode[@]}"
# 		do
# 		    for d in "${drivers[@]}"
# 			do
# 				for l in "${lanes[@]}"
# 				do
for c in "${cars[@]}"
do
    for g in "${gaps[@]}"
	do
		exp "$c" "$g" &
	done
done
# 				done
# 			done
# 		done
# 	done
# done

# find -name "* *" -type d | rename 's/ //g' && find -name "*MPC*" -type d | rename 's/,/-/g' && find -name "*MPC*" -type d | rename 's/\(/_/g' && find -name "*MPC*" -type d | rename 's/\)//g'
