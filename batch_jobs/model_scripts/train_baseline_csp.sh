#!/bin/bash

root_dir='/home/nschiou2/mindportal/'

cd $root_dir

windows=('all' 0 1 2 3 4 5 6 7)
selection_methods=('PCA' 'MI' 'None')
n_filters=(16 8 4 2)

for i in ${!n_filters[@]}; do
for j in ${!windows[@]}; do
for k in ${!selection_methods[@]}; do

  dirname='linear_svm_'${windows[$j]}

  if [[ "${selection_methods[$k]}" == "None" ]]
  then
    dirname='no_select_'$dirname
    log_var_dirname=$dirname'_log_var'
  elif [[ "${selection_methods[$k]}" == "PCA" ]]
  then
    dirname=$dirname'_pca'
  else
    dirname=$dirname'_mi'
  fi

  echo "pipenv run python ./base_classifier/train_base_clf.py \
        --exp_dir ${n_filters[$i]}_filters/$dirname \
        --csp \
        --clf_type SVM \
        --window ${windows[$j]} \
        --anchor pc \
        --selection_method ${selection_methods[$k]} \
        --n_components $((${n_filters[$i]} / 2)) \
        --classification_task motor_LR \
        --n_filters ${n_filters[$i]} \
        --max_iter 1000000 \
        --n_splits 5;"

  pipenv run python ./base_classifier/train_base_clf.py \
  --exp_dir ${n_filters[$i]}_filters/$dirname \
  --csp \
  --clf_type SVM \
  --window ${windows[$j]} \
  --anchor pc \
  --selection_method ${selection_methods[$k]} \
  --n_components $((${n_filters[$i]} / 2)) \
  --classification_task motor_LR \
  --n_filters ${n_filters[$i]} \
  --max_iter 1000000 \
  --n_splits 5;

  if [[ "${selection_methods[$k]}" == "None" ]]
  then
    echo "pipenv run python ./base_classifier/train_base_clf.py \
          --exp_dir ${n_filters[$i]}_filters/$log_var_dirname \
          --csp \
          --clf_type SVM \
          --window ${windows[$j]} \
          --anchor pc \
          --log_variance_feats \
          --selection_method ${selection_methods[$k]} \
          --n_components $((${n_filters[$i]} / 2)) \
          --classification_task motor_LR \
          --n_filters ${n_filters[$i]} \
          --max_iter 1000000 \
          --n_splits 5;"

    pipenv run python ./base_classifier/train_base_clf.py \
    --exp_dir ${n_filters[$i]}_filters/$log_var_dirname \
    --csp \
    --clf_type SVM \
    --window ${windows[$j]} \
    --anchor pc \
    --log_variance_feats \
    --selection_method ${selection_methods[$k]} \
    --n_components $((${n_filters[$i]} / 2)) \
    --classification_task motor_LR \
    --n_filters ${n_filters[$i]} \
    --max_iter 1000000 \
    --n_splits 5;
  fi

done
done
done