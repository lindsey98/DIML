dataset=${1:-cub200}
bs=${2:-32}
loss=${3:-margin_diml}
epochs=${4:-50}
seed=${5:-0}
IPC=${6:-8}
nb_workers=${7:-4}

python train_diml.py --dataset $dataset --loss $loss --batch_mining distance \
              --group ${dataset}_$loss --seed $seed \
              --bs $bs --data_sampler class_random --samples_per_class 4 \
              --arch resnet50_diml_frozen_normalize  --n_epochs $epochs \
              --lr 0.00001 --embed_dim 512 --evaluate_on_gpu --IPC $IPC \
              --nb_workers $nb_workers
