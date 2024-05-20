## ISCF

This repo is based on [cl-lite](https://github.com/gqk/cl-lite).

## Experiment

- Install dependencies

    ```shell
    pip install -r requirements.txt
    ```
- Prepare datasets

    1. create a dataset root diretory, e.g., data
    2. cifar100 will be automatically downloaded
    3. download and unzip [tiny-imagenet200](http://cs231n.stanford.edu/tiny-imagenet-200.zip) to dataset root diretory
    4. follow [PODNet](https://github.com/arthurdouillard/incremental_learning.pytorch/tree/master/imagenet_split) to prepare imagenet100 dataset
    5. the overview of dataset root diretory

        ```shell
        ├── cifar100
        │   └── cifar-100-python
        ├── imagenet100
        │   ├── train
        │   ├── train_100.txt
        │   ├── val
        │   └── val_100.txt
        └── imagenet1000
            ├── train
            ├── val
            ├── train_1000.txt
            └── val_1000.txt
        ```

- Run experiment
    ```shell
    python main.py --config config/cifar-100/cifar100_iscf.yaml
    ```

