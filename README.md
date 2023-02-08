# X-GOAL: Multiplex Heterogeneous Graph Prototypical Contrastive Learning
This is the PyTorch implementation of the paper:

[X-GOAL: Multiplex Heterogeneous Graph Prototypical Contrastive Learning](https://arxiv.org/abs/2109.03560), CIKM'2022\
Baoyu Jing, Shengyu Feng, Yuejia Xiang, Xi Chen, Yu Chen and Hanghang Tong

## Requirements
- Python=3.10
- numpy=1.23.5
- scipy=1.9.3
- scikit-learn=0.22.0
- tqdm=4.64.1
- torch=1.13.0 

Packages can be installed via: `pip install -r requirements.txt`.
For PyTorch, please install the version compatible with your machine.


## Data
The pre-processed data can be downloaded from [here](https://www.dropbox.com/s/48oe7shjq0ih151/data.tar.gz?dl=0). 
Please put the pre-processed data under the folder `data`.
Each pre-processed dataset is a dictionary containing the following keys:
- `train_idx`, `val_idx` and `test_idx` are indices for training, validation and testing; 
`label` corresponds to the labels of the nodes;
- the layer names of the dataset: e.g., `MAM` and `MDM` for the `imdb` dataset.

## Run
1. Download the pre-processed data from [here](https://www.dropbox.com/s/48oe7shjq0ih151/data.tar.gz?dl=0)
   and put it to the folder `data`.
2. Specify the arguments in the `xgoal_{datasetname}.py`.
3. Run the code by `python xgoal_{datasetname}.py`.
4. `goal_example.py` is an example file for the GOAL model.


## Citation
Please cite the following paper, if you find the repository or the paper useful.

[X-GOAL: Multiplex Heterogeneous Graph Prototypical Contrastive Learning](https://arxiv.org/abs/2109.03560), CIKM'2022\
Baoyu Jing, Shengyu Feng, Yuejia Xiang, Xi Chen, Yu Chen and Hanghang Tong

```
@article{jing2021x,
  title={X-GOAL: Multiplex Heterogeneous Graph Prototypical Contrastive Learning},
  author={Jing, Baoyu and Feng, Shengyu and Xiang, Yuejia and Chen, Xi and Chen, Yu and Tong, Hanghang},
  journal={arXiv preprint arXiv:2109.03560},
  year={2021}
}
```