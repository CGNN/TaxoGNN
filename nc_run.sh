#!/usr/bin/env bash

export PYTHONPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

#CUDA_VISIBLE_DEVICES=0 python nc_evaluation.py --method graphsage &
#sleep 10
CUDA_VISIBLE_DEVICES=1 python nc_evaluation.py --method dgi &
sleep 10
#CUDA_VISIBLE_DEVICES=2 python nc_evaluation.py --method taxognn &
#sleep 10
#CUDA_VISIBLE_DEVICES=3 python nc_evaluation.py --method orig_graphsage &
#sleep 10
#CUDA_VISIBLE_DEVICES=4 python nc_evaluation.py --method bert &
#sleep 10

#CUDA_VISIBLE_DEVICES=5 python nc_evaluation.py --method graphsage &
#sleep 10
CUDA_VISIBLE_DEVICES=7 python nc_evaluation.py --method dgi &
sleep 10
#CUDA_VISIBLE_DEVICES=7 python nc_evaluation.py --method taxognn &
#sleep 10
#CUDA_VISIBLE_DEVICES=0 python nc_evaluation.py --method orig_graphsage &
#sleep 10
#CUDA_VISIBLE_DEVICES=1 python nc_evaluation.py --method bert &
#sleep 10

#CUDA_VISIBLE_DEVICES=2 python nc_evaluation.py --method graphsage &
#sleep 10
CUDA_VISIBLE_DEVICES=7 python nc_evaluation.py --method dgi &
sleep 10
#CUDA_VISIBLE_DEVICES=4 python nc_evaluation.py --method taxognn &
#sleep 10
#CUDA_VISIBLE_DEVICES=5 python nc_evaluation.py --method orig_graphsage &
#sleep 10
#CUDA_VISIBLE_DEVICES=6 python nc_evaluation.py --method bert &


#CUDA_VISIBLE_DEVICES=7 python nc_evaluation.py --method graphsage &
#sleep 10
CUDA_VISIBLE_DEVICES=7 python nc_evaluation.py --method dgi &
sleep 10
#CUDA_VISIBLE_DEVICES=5 python nc_evaluation.py --method taxognn &
#sleep 10
#CUDA_VISIBLE_DEVICES=6 python nc_evaluation.py --method orig_graphsage &
#sleep 10
#CUDA_VISIBLE_DEVICES=1 python nc_evaluation.py --method bert &
#sleep 10


#CUDA_VISIBLE_DEVICES=0 python nc_evaluation.py --method graphsage &
#sleep 10
CUDA_VISIBLE_DEVICES=3 python nc_evaluation.py --method dgi &
sleep 10
#CUDA_VISIBLE_DEVICES=2 python nc_evaluation.py --method taxognn &
#sleep 10
#CUDA_VISIBLE_DEVICES=3 python nc_evaluation.py --method orig_graphsage &
#sleep 10
#CUDA_VISIBLE_DEVICES=4 python nc_evaluation.py --method bert &
#sleep 10
#
#CUDA_VISIBLE_DEVICES=5 python nc_evaluation.py --method graphsage &
#sleep 10
CUDA_VISIBLE_DEVICES=3 python nc_evaluation.py --method dgi &
sleep 10
#CUDA_VISIBLE_DEVICES=7 python nc_evaluation.py --method taxognn &
#sleep 10
#CUDA_VISIBLE_DEVICES=0 python nc_evaluation.py --method orig_graphsage &
#sleep 10
#CUDA_VISIBLE_DEVICES=1 python nc_evaluation.py --method bert &
#
#sleep 10
#CUDA_VISIBLE_DEVICES=2 python nc_evaluation.py --method graphsage &
#sleep 10
CUDA_VISIBLE_DEVICES=1 python nc_evaluation.py --method dgi &
sleep 10
#CUDA_VISIBLE_DEVICES=4 python nc_evaluation.py --method taxognn &
#sleep 10
#CUDA_VISIBLE_DEVICES=5 python nc_evaluation.py --method orig_graphsage &
#sleep 10
#CUDA_VISIBLE_DEVICES=6 python nc_evaluation.py --method bert &
#sleep 10
#
#CUDA_VISIBLE_DEVICES=7 python nc_evaluation.py --method graphsage &
#sleep 10
CUDA_VISIBLE_DEVICES=1 python nc_evaluation.py --method dgi &
sleep 10
#CUDA_VISIBLE_DEVICES=5 python nc_evaluation.py --method taxognn &
#sleep 10
#CUDA_VISIBLE_DEVICES=6 python nc_evaluation.py --method orig_graphsage &
#sleep 10
#CUDA_VISIBLE_DEVICES=1 python nc_evaluation.py --method bert &
#sleep 10
#
#
#
#
#CUDA_VISIBLE_DEVICES=0 python nc_evaluation.py --method graphsage &
#sleep 10
CUDA_VISIBLE_DEVICES=5 python nc_evaluation.py --method dgi &
sleep 10
#CUDA_VISIBLE_DEVICES=2 python nc_evaluation.py --method taxognn &
#sleep 10
#CUDA_VISIBLE_DEVICES=3 python nc_evaluation.py --method orig_graphsage &
#sleep 10
#CUDA_VISIBLE_DEVICES=4 python nc_evaluation.py --method bert &
#sleep 10
#
#CUDA_VISIBLE_DEVICES=5 python nc_evaluation.py --method graphsage &
#sleep 10
CUDA_VISIBLE_DEVICES=3 python nc_evaluation.py --method dgi &
sleep 10
#CUDA_VISIBLE_DEVICES=7 python nc_evaluation.py --method taxognn &
#sleep 10
#CUDA_VISIBLE_DEVICES=0 python nc_evaluation.py --method orig_graphsage &
#sleep 10
#CUDA_VISIBLE_DEVICES=1 python nc_evaluation.py --method bert &



