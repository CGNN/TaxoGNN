#!/usr/bin/env bash
export PYTHONPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

id=0
CUDA_VISIBLE_DEVICES=2 python graphSAGE/main_graphsage.py --dataset MAG --mode node --out_features 128 --epochs 30 --orig 0 --gat 1 --save_id $id &



#id=1
#CUDA_VISIBLE_DEVICES=1 python graphSAGE/unsup_main_graphsage.py --dataset MeSH --mode link --out_features 128 --epochs 5 --orig 0 --save_id $id &

#id=2
#CUDA_VISIBLE_DEVICES=2 python graphSAGE/unsup_main_graphsage.py --dataset MeSH --mode link --out_features 128 --epochs 5 --orig 0 --save_id $id
#
#
#id=3
#CUDA_VISIBLE_DEVICES=2 python graphSAGE/unsup_main_graphsage.py --dataset MeSH --mode link --out_features 128 --epochs 5 --orig 0 --save_id $id
#
#
#id=4
#CUDA_VISIBLE_DEVICES=2 python graphSAGE/unsup_main_graphsage.py --dataset MeSH --mode link --out_features 128 --epochs 5 --orig 0 --save_id $id


#id=1
#CUDA_VISIBLE_DEVICES=3 python graphSAGE/main_graphsage.py --dataset MeSH --mode link --out_features 128 --epochs 30 --orig 0 --save_id $id &
#sleep 10
#CUDA_VISIBLE_DEVICES=4 python taxo_prop.py --dataset MeSH --task link --out_features 128 --epochs 60 --save_id $id &
#sleep 10
#CUDA_VISIBLE_DEVICES=5 python graphSAGE/unsup_main_graphsage.py --dataset MeSH --mode link --out_features 128 --epochs 15 --orig 0 --save_id $id &

#id=2
#CUDA_VISIBLE_DEVICES=0 python graphSAGE/main_graphsage.py --dataset MeSH --mode link --out_features 128 --epochs 30 --orig 0 --save_id $id &
#sleep 10
#CUDA_VISIBLE_DEVICES=0 python taxo_prop.py --dataset MeSH --task link --out_features 128 --epochs 60 --save_id $id &
#sleep 10
#CUDA_VISIBLE_DEVICES=7 python graphSAGE/unsup_main_graphsage.py --dataset MeSH --mode link --out_features 128 --epochs 15 --orig 0 --save_id $id &


#id=3
#CUDA_VISIBLE_DEVICES=2 python graphSAGE/main_graphsage.py --dataset MeSH --mode link --out_features 128 --epochs 30 --orig 0 --save_id $id &
#sleep 10
#CUDA_VISIBLE_DEVICES=3 python taxo_prop.py --dataset MeSH --task link --out_features 128 --epochs 60 --save_id $id &
#sleep 10
#CUDA_VISIBLE_DEVICES=4 python graphSAGE/unsup_main_graphsage.py --dataset MeSH --mode link --out_features 128 --epochs 15 --orig 0 --save_id $id &


#id=4
#CUDA_VISIBLE_DEVICES=4 python graphSAGE/main_graphsage.py --dataset MeSH --mode link --out_features 128 --epochs 30 --orig 0 --save_id $id &
#sleep 10
#CUDA_VISIBLE_DEVICES=0 python taxo_prop.py --dataset MeSH --task link --out_features 128 --epochs 60 --save_id $id &
#sleep 10
#CUDA_VISIBLE_DEVICES=6 python graphSAGE/unsup_main_graphsage.py --dataset MeSH --mode link --out_features 128 --epochs 15 --orig 0 --save_id $id &



#CUDA_VISIBLE_DEVICES=2 python accurate_nn.py --method bert &
#sleep 10
#CUDA_VISIBLE_DEVICES=3 python accurate_nn.py --method taxognn &
#sleep 10
#CUDA_VISIBLE_DEVICES=5 python accurate_nn.py --method graphsage_ssl &
#sleep 10
#CUDA_VISIBLE_DEVICES=7 python accurate_nn.py --method graphsage_unsup &



#CUDA_VISIBLE_DEVICES=2 python lp_evaluation.py --method bert >> ./txt_logs/bert_lp_evaluation.txt &
#sleep 10
#CUDA_VISIBLE_DEVICES=4 python lp_evaluation.py --method taxognn >> ./txt_logs/taxognn_lp_evaluation.txt &
#sleep 10
#CUDA_VISIBLE_DEVICES=5 python lp_evaluation.py --method graphsage_ssl >> ./txt_logs/graphsage_ssl_lp_evaluation.txt  &
#sleep 10
#CUDA_VISIBLE_DEVICES=7 python lp_evaluation.py --method graphsage_unsup >> ./txt_logs/graphsage_unsup_lp_evaluation.txt &