# Channel Pruning with XXX Grouping for Efficient Medical Image Segmentation

## Installation

1. Clone the repository:
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Datasets

Ensure the datasets are prepared and stored in the correct directories as specified in the `config.py` file.

## Training

To train the model, run the following command:

```bash
python train.py --dataset <dataset_name> --arch <model_architecture> --epochs <num_epochs> --batch_size <batch_size>
```

### Example:

```bash
python train.py --dataset isic --arch unet --epochs 50 --batch_size 16
```

## Validation: 

```bash 
python val.py --dataset <dataset_name> --arch <model_architecture> --batch_size <batch_size> --model_path <path_to_model>
```

### Example:

```bash
python val.py --dataset isic --arch unet --batch_size 1 --model_path ./models/best_model.pth
```

## Configuration:

You can adjust various hyperparameters and settings in the `config.py` file, including: 

- Dataset name 
- Image file extension 
- Mask file extension 
- Optimizer 
- Learning rate 
- Batch size 
- Number of epochs

## Files Description:

- `archs.py`: Defines the neural network architectures. 
- `config.py`: Contains configuration settings. 
- `dataset.py`: Handles dataset loading and preprocessing. 
- `losses.py`: Implements loss functions. 
- `metrics.py`: Contains evaluation metrics. 
- `train.py`: Script for training the model. 
- `utils.py`: Utility functions. 
- `val.py`: Script for validating the model. 



