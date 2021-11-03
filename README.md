# TaxoGNN
Source code for TaxoGNN

## Training Mode
link: both keywords and taxonomy labels

node: hide the taxonomy information for classification

## Train TaxoGNN Embedding
```
python taxo_prop.py --dataset MAG --task link --epochs 40
```

## Evaluate Link Prediction
```
python lp_evaluation.py --dataset MAG --method taxognn
```

## Evaluate Node Classification
```
python nc_evaluation.py --dataset MAG --method taxognn
```
