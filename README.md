# CIFARTile
A three-stage CNN model for CIFARTile label prediction

### Model
a pre-trained model under the `models` folder, and the structure of it could be demonstrate detailedly by using the wonderful tool ([Netron](https://netron.app)).

## Training Steps
1. run `CIFARTile/train_cifar10.py` with specific backbone set by `--backbone`, then get the model weight file saved in `checkpoint_file`
2. run `CIFARTile/main.py` with pre-trained weight of backbone set by 'cifar_weight' and the backbone type set by `backbone`, then get the final model weight
3. (optional) run `CIFARTile/eval_model.py` to calculate prediction accuracy and export the model with network structure inside (visiable with **Netron**)
