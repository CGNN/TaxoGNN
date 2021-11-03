#!/usr/bin/env bash
export PYTHONPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# baseline
#CUDA_VISIBLE_DEVICES=5 python taxo_prop.py --dataset MAG --task node --epochs 40 --log_type txt &


#CUDA_VISIBLE_DEVICES=0 python graphSAGE/main_graphsage.py --dataset MAG --mode link --out_features 128 --epochs 20 --orig 0 --save_id 0 &
#sleep 10
#CUDA_VISIBLE_DEVICES=1 python graphSAGE/main_graphsage.py --dataset MAG --mode node --out_features 128 --epochs 20 --orig 0 --save_id 0 &
#sleep 10


#CUDA_VISIBLE_DEVICES=2 python graphSAGE/main_graphsage.py --dataset MAG --mode link --out_features 128 --epochs 20 --orig 0 --save_id 1 &
#sleep 10
#CUDA_VISIBLE_DEVICES=3 python graphSAGE/main_graphsage.py --dataset MAG --mode node --out_features 128 --epochs 20 --orig 0 --save_id 1 &
#sleep 10


CUDA_VISIBLE_DEVICES=1 python graphSAGE/main_graphsage.py --dataset MAG --mode link --out_features 128 --epochs 20 --orig 0 --save_id 2 &
sleep 10
#CUDA_VISIBLE_DEVICES=5 python graphSAGE/main_graphsage.py --dataset MAG --mode node --out_features 128 --epochs 20 --orig 0 --save_id 2 &
#sleep 10


#CUDA_VISIBLE_DEVICES=6 python graphSAGE/main_graphsage.py --dataset MAG --mode link --out_features 128 --epochs 20 --orig 0 --save_id 3 &
#sleep 10
#CUDA_VISIBLE_DEVICES=7 python graphSAGE/main_graphsage.py --dataset MAG --mode node --out_features 128 --epochs 20 --orig 0 --save_id 3 &
#sleep 10


#<<<<<<< HEAD
#CUDA_VISIBLE_DEVICES=2 python graphSAGE/main_graphsage.py --dataset MAG --mode link --out_features 128 --epochs 20 --orig 0 --save_id 4 &
#sleep 10
#CUDA_VISIBLE_DEVICES=3 python graphSAGE/main_graphsage.py --dataset MAG --mode node --out_features 128 --epochs 20 --orig 0 --save_id 4 &
#=======
## for node classification == DGI
python ./DGI/main_mag.py --mode node --gpu 3 --save_id 1



# supervision link prediction
python sup-link/main.py --gpu 5 --method dgi --K 6 --epoch_num 10 --neg_train 1 > MAG_dgi_sup_link.txt
#>>>>>>> b4339a2d2b18f9544c26beaaabc896c6ec94583d
