#!/bin/bash

root_dir='/home/nschiou2/mindportal/'

cd $root_dir

windows=('all' 'rt' 'pre_stim' 'init' 'pre_rt' 'post_rt')
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

  echo "pipenv run python ./base_classifier/train_base_clf_csp.py \
        --exp_dir ${n_filters[$i]}_filters/$dirname \
        --clf_type SVM \
        --window ${windows[$j]} \
        --selection_method ${selection_methods[$k]} \
        --n_components $((${n_filters[$i]} / 2)) \
        --classification_type motor_LR \
        --n_filters ${n_filters[$i]} \
        --max_iter 1000000 \
        --n_splits 5;"

  pipenv run python ./base_classifier/train_base_clf_csp.py \
  --exp_dir ${n_filters[$i]}_filters/$dirname \
  --clf_type SVM \
  --window ${windows[$j]} \
  --selection_method ${selection_methods[$k]} \
  --n_components $((${n_filters[$i]} / 2)) \
  --classification_type motor_LR \
  --n_filters ${n_filters[$i]} \
  --max_iter 1000000 \
  --n_splits 5;

  if [[ "${selection_methods[$k]}" == "None" ]]
  then
    echo "pipenv run python ./base_classifier/train_base_clf_csp.py \
          --exp_dir ${n_filters[$i]}_filters/$log_var_dirname \
          --clf_type SVM \
          --window ${windows[$j]} \
          --log_variance_feats \
          --selection_method ${selection_methods[$k]} \
          --n_components $((${n_filters[$i]} / 2)) \
          --classification_type motor_LR \
          --n_filters ${n_filters[$i]} \
          --max_iter 1000000 \
          --n_splits 5;"

    pipenv run python ./base_classifier/train_base_clf_csp.py \
    --exp_dir ${n_filters[$i]}_filters/$log_var_dirname \
    --clf_type SVM \
    --window ${windows[$j]} \
    --log_variance_feats \
    --selection_method ${selection_methods[$k]} \
    --n_components $((${n_filters[$i]} / 2)) \
    --classification_type motor_LR \
    --n_filters ${n_filters[$i]} \
    --max_iter 1000000 \
    --n_splits 5;
  fi

done
done
done