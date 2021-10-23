#!/bin/bash

root_dir='/home/nschiou2/mindportal/'

cd $root_dir

windows=('all' 'rt' 'pre_stim' 'init' 'pre_rt' 'post_rt')
selection_methods=('PCA' 'MI' 'None')

for i in ${!windows[@]}; do
for j in ${!selection_methods[@]}; do

  dirname='linear_svm_'${windows[$i]}

  if [[ "${selection_methods[$j]}" == "None" ]]
  then
    dirname='no_select_'$dirname
  elif [[ "${selection_methods[$j]}" == "PCA" ]]
  then
    dirname=$dirname'_pca'
  else
    dirname=$dirname'_mi'
  fi

  echo "pipenv run python ./base_classifier/train_base_clf.py \
        --exp_dir $dirname \
        --clf_type SVM \
        --window ${windows[$i]} \
        --selection_method ${selection_methods[$j]} \
        --classification_type motor_LR \
        --max_iter 1000000 \
        --n_splits 5;"

  pipenv run python ./base_classifier/train_base_clf.py \
  --exp_dir $dirname \
  --clf_type SVM \
  --window ${windows[$i]} \
  --selection_method ${selection_methods[$j]} \
  --classification_type motor_LR \
  --max_iter 1000000 \
  --n_splits 5;

done
done