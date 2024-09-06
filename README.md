# TransLoc4D: Transformer-based 4D Radar Place Recognition

## Abstract
Place recognition is crucial for unmanned vehicles interms of localization and mapping. Recent years have witnessed numerous explorations in the field, where 2D cameras and 3D LiDARs are mostly employed. Despite theiradmirable performance, they may encounter challenges inadverse weather such as rain and fog. Hopefully, 4D millimeter-wave radar emerges as a promising alternative, as its longer wavelength makes it virtually immune to interference from tiny particles of fog and rain. Therefore, in this work, we propose a novel 4D radar place recognition model, TransLoc4D, based on sparse convolutions andTransformer structures. Specifically, a MinkLoc4D backbone is first proposed to leverage the multimodal information from 4D radar scans. Rather than merely capturing geometric structures of point clouds, MinkLoc4D additionally explores their intensity and velocity properties. After feature extraction, a Transformer layer is introducedto enhance local features before aggregation, where linear self-attention captures the long-range dependencies of thepoint cloud, alleviating its sparsity and noise. To validate TransLoc4D, we construct two datasets and set up benchmarks for 4D radar place recognition. Experiments validate the feasibility of TransLoc4D and demonstrate it canrobustly deal with dynamic and adverse environments.

## Installation
### 1. Using Docker
For ease of setup, use the Docker-compose file or the `.devcontainer` configuration with VSCode's Remote - Containers extension. Before running, ensure to modify the `docker-compose.yml` file to correctly map the paths to the datasets.

### 2. Manual Installation
Alternatively, perform a manual installation as follows:
1. Install PyTorch:
   Ensure that you install the correct version of PyTorch that is compatible with your CUDA version to leverage GPU acceleration. Visit the [PyTorch official website](https://pytorch.org/get-started/locally/) for the command tailored to your environment, or use a general installation command like:
   ```
   pip install torch torchvision torchaudio
   ```
   
2. Clone and install MinkowskiEngine:
   ```
   git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine"
   cd MinkowskiEngine
   python setup.py install --force_cuda --blas=openblas
   ```

3. Install other required Python packages:
   ```
   cd {Project Folder}
   pip install -r requirements.txt
   pip install -e .
   ```

## Datasets and Model Weights
The datasets used in this study (dynamic points removed with velocity) can be found [here](https://entuedu-my.sharepoint.com/:f:/g/personal/heshan001_e_ntu_edu_sg/EtDhy41KPAFFqTD6wb9-1EABl-bXqnez8HXFRJgIPOJosg?e=IRzzft), and model weights trained on the NTU dataset are available [here](https://entuedu-my.sharepoint.com/:f:/g/personal/heshan001_e_ntu_edu_sg/EpNaXwtTW99KghTTzrkbisEBQX5wDyVvmIjRBQZF8e7_Kg?e=55Qpup).

### Model Performance

| Test Set     | nyl-night | nyl-rain | src-night | sjtu4dpr-testa | sjtu4dpr-testb |
| ------------ | --------- | -------- | --------- | -------------- | -------------- |
| Recall@1 (%) | 97.1      | 86.8     | 94.5      | 90.8           | 85.9           |



## Usage
To prepare datasets for training and evaluation, run the following scripts:
- For training set generation, modify path and config in `scripts/generate_trainset.py` and then run: 
  ```
  python scripts/generate_trainset.py
  ```
- For test set generation, modify path and config in `scripts/generate_testsets.py`:
  ```
  python scripts/generate_testsets.py
  ```

### Running Evaluation and Training
#### Evaluation
To evaluate the model, execute the following command:
```
python scripts/eval.py --database_pickle "/path/to/database_pickle" --query_pickle "/path/to/query_pickle" --model_config "config/model/transloc4d.txt" --weights "path/to/weights"
```
**Parameters:**
- `--database_pickle`: Path to the database pickle file.
- `--query_pickle`: Path to the query pickle file.
- `--model_config`: Path to the model-specific configuration file.
- `--weights`: Path to the trained model weights.

#### Training
To train the model, run the following command:
```
python scripts/train.py --config "config/train/ntu-rsvi.txt" --model_config "config/model/transloc4d.txt"
```
**Parameters:**
- `--config`: Path to the training configuration file.
- `--model_config`: Path to the model-specific configuration file.

Training result can then be found in `weights`.

## Citation
If you find this work useful, please cite our paper:
```
@inproceedings{peng2024transloc4d,
  title={TransLoc4D: Transformer-based 4D Radar Place Recognition},
  author={Peng, Guohao and Li, Heshan and Zhao, Yangyang and Zhang, Jun and Wu, Zhenyu and Zheng, Pengyu and Wang, Danwei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={17595--17605},
  year={2024}
}
```
## 
## Acknowledgement

Our code is based on [PointNetVLAD](https://github.com/mikacuy/pointnetvlad), [MinkLoc3Dv2](https://github.com/jac99/MinkLoc3Dv2), [PPT-Net](https://github.com/fpthink/PPT-Net) and [PTC-Net](https://github.com/LeegoChen/PTC-Net).
