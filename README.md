# DIML

This code contains the implementation for paper **Towards Interpretable Deep Metric Learning with Structural Matching**.

![intro](figs/intro.gif)

## Usage
### Requirement
- python3
- PyTorch 1.7

### Dataset Preparation
Please follow the instruction in [RevisitDML](https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch) to download the datasets and put all the datasets in `data` folder.

### Training & Evaluation
To train the baseline models, run the scripts in `scripts/baselines`. For example:
```bash
./script/baselines/cub_runs.sh
```

To test the baseline models with our proposed DIML, first edit the checkpoint path in `test_diml.py`, then run
```bash
./scripts/diml/test_diml.sh
```

You can also incorporate DIML into the training objectives. We provide two examples which apply DIML to  Margin and Multi-Similarity loss. To train DIML models, run
```bash
# ./scripts/diml/train_diml.sh <gpu_id> <dataset> <batch_size> <loss> <num_epochs>
# where loss could be margin_diml or multisimilarity_diml
# e.g.
./scripts/diml/train_diml.sh 0 cub200 112 margin_diml 150
```

## Acknowledgement
The code is based on [RevisitDML](https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch).
