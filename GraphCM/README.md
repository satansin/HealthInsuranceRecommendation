## A Graph-Enhanced Click Model for Web Search (GraphCM)

**NOTE**: !!DO not run run.sh directly, check run.sh to see how to run (after reading this file)!!

### Installing the environment (tested in conda win)

1. Open/install anaconda:
Anaconda Prompt (Miniconda3)

2. List out the environments:
```conda info --envs```
or
```conda env list```

3. If there exists "iprotect_graphcm", activate it and go to 6:
```conda activate iprotect_graphcm```
Otherwise, go to 4

4. Create the environment and activate it:
```
conda create --name iprotect_graphcm python=3.7
conda activate iprotect_graphcm
```

5. Follow the guidance to install other independencies:

The versions of torch-cluster, torch-scatter, torch-sparse, torch-spline-conv are strictly required for torch-geometric package. You can follow the installation instruction in the PyG official website: torch-geometric 1.6.3.
- python 3.7
- pytorch 1.6.0+cu101
- torchvision 0.7.0+cu101
- torch-cluster 1.5.8
- torch-scatter 2.0.5
- torch-sparse 0.6.8
- torch-spline-conv 1.2.0
- torch-geometric 1.6.3
- tensorboardx 2.1

Follow: https://pytorch.org/get-started/previous-versions/
```
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
```
Note that torchvision 0.7.0+cu101 is automatically included.

```
pip install torch-cluster==1.5.8 -f https://pytorch-geometric.com/whl/torch-1.6.0+cu101.html
pip install torch-scatter==2.0.5 -f https://pytorch-geometric.com/whl/torch-1.6.0+cu101.html
pip install torch-sparse==0.6.8 -f https://pytorch-geometric.com/whl/torch-1.6.0+cu101.html
pip install torch-spline-conv==1.2.0 -f https://pytorch-geometric.com/whl/torch-1.6.0+cu101.html
pip install torch-geometric==1.6.3 -f https://pytorch-geometric.com/whl/torch-1.6.0+cu101.html
pip install tensorboardx==2.1
```

6. Go to the working folder, try with:
```
python -u run.py --rank --optim adam --eval_freq 100 --check_point 100 --dataset demo --combine exp_mul --gnn_neigh_sample 0 --gnn_concat False --inter_neigh_sample 0 --learning_rate 0.001 --lr_decay 0.5 --weight_decay 1e-5 --dropout_rate 0.5 --num_steps 20000 --embed_size 64 --hidden_size 64 --batch_size 256 --patience 6 --model_dir ./GraphCM/models/ --result_dir ./GraphCM/results/ --summary_dir ./GraphCM/summary/ --log_dir ./GraphCM/log/ --load_model=20000
```
or
```
python -u run.py --train --optim adam --eval_freq 100 --check_point 1000 --dataset demo --combine exp_mul --gnn_neigh_sample 0 --gnn_concat False --inter_neigh_sample 0 --learning_rate 0.001 --lr_decay 0.5 --weight_decay 1e-5 --dropout_rate 0.5 --num_steps 20000 --embed_size 64 --hidden_size 64 --batch_size 256 --patience 6 --model_dir ./GraphCM/models/ --result_dir ./GraphCM/results/ --summary_dir ./GraphCM/summary/ --log_dir ./GraphCM/log/
```

Possible error 1:
```
Traceback (most recent call last):
  File "run.py", line 10, in <module>
    from model import Model
  File "C:\Users\hliubs\Desktop\GraphCM-main\model.py", line 9, in <module>
    from tensorboardX import SummaryWriter
  File "C:\Users\hliubs\.conda\envs\iprotect_graphcm\lib\site-packages\tensorboardX\__init__.py", line 5, in <module>
    from .torchvis import TorchVis
  File "C:\Users\hliubs\.conda\envs\iprotect_graphcm\lib\site-packages\tensorboardX\torchvis.py", line 11, in <module>
    from .writer import SummaryWriter
  File "C:\Users\hliubs\.conda\envs\iprotect_graphcm\lib\site-packages\tensorboardX\writer.py", line 16, in <module>
    from .event_file_writer import EventFileWriter
  File "C:\Users\hliubs\.conda\envs\iprotect_graphcm\lib\site-packages\tensorboardX\event_file_writer.py", line 28, in <module>
    from .proto import event_pb2
  File "C:\Users\hliubs\.conda\envs\iprotect_graphcm\lib\site-packages\tensorboardX\proto\event_pb2.py", line 16, in <module>
    from tensorboardX.proto import summary_pb2 as tensorboardX_dot_proto_dot_summary__pb2
  File "C:\Users\hliubs\.conda\envs\iprotect_graphcm\lib\site-packages\tensorboardX\proto\summary_pb2.py", line 16, in <module>
    from tensorboardX.proto import tensor_pb2 as tensorboardX_dot_proto_dot_tensor__pb2
  File "C:\Users\hliubs\.conda\envs\iprotect_graphcm\lib\site-packages\tensorboardX\proto\tensor_pb2.py", line 16, in <module>
    from tensorboardX.proto import resource_handle_pb2 as tensorboardX_dot_proto_dot_resource__handle__pb2
  File "C:\Users\hliubs\.conda\envs\iprotect_graphcm\lib\site-packages\tensorboardX\proto\resource_handle_pb2.py", line 42, in <module>
    serialized_options=None, file=DESCRIPTOR),
  File "C:\Users\hliubs\.conda\envs\iprotect_graphcm\lib\site-packages\google\protobuf\descriptor.py", line 560, in __new__
    _message.Message._CheckCalledFromGeneratedFile()
TypeError: Descriptors cannot not be created directly.
If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0.
If you cannot immediately regenerate your protos, some other possible workarounds are:
 1. Downgrade the protobuf package to 3.20.x or lower.
 2. Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (but this will use pure-Python parsing and will be much slower).
```
SOLUTION:
```pip install --upgrade "protobuf<=3.20.1"```


=========================================================================

Below are steps to install dependencies of the newest versions

1. Install pytorch-CUDA with pip (note that by default pytorch will install CPU version)
```https://pytorch.org/get-started/locally/```
Or: google search install pytorch (would be better in virtual environment)

2. Follow the guidance to install other independencies similarly


## (Below are the original README) A Graph-Enhanced Click Model for Web Search (GraphCM)

### Introduction

This is the pytorch implementation of GraphCM proposed in the paper: [A Graph-Enhanced Click Model for Web Search. SIGIR 2021](https://dl.acm.org/doi/10.1145/3404835.3462895).

### Requirements

**NOTE**: The versions of torch-cluster, torch-scatter, torch-sparse, torch-spline-conv are strictly required for torch-geometric package. You can follow the installation instruction in the PyG official website: [torch-geometric 1.6.3](https://pytorch-geometric.readthedocs.io/en/1.6.3/).

- python 3.7
- pytorch 1.6.0+cu101
- torchvision 0.7.0+cu101
- torch-cluster 1.5.8
- torch-scatter 2.0.5
- torch-sparse 0.6.8
- torch-spline-conv 1.2.0
- torch-geometric 1.6.3
- tensorboardx 2.1

### Input Data Formats

After data pre-processing, we can put all the generated files into ```./data/dataset/``` folder as input files for GraphCM. Demo input files are available under the ```./data/demo/``` directory. 

The format of train & valid & test & label input files is as follows:

- Each line: ```<session id><tab><query id><tab>[<document ids>]<tab>[<vtype ids>]<tab>[<clicks infos>]<tab>[<relevance>]```

### Quick Start

We provide quick start command in ```./run.sh```. Note that input files that are related to graph modules are not provided in this repo. You can genenrate graph-related input files using data preprocess files in the ```./data_preprocess/``` fold.

### Citation

If you find the resources in this repo useful, please cite our work.

```
@inproceedings{lin2021graph,
  title={A Graph-Enhanced Click Model for Web Search},
  author={Lin, Jianghao and Liu, Weiwen and Dai, Xinyi and Zhang, Weinan and Li, Shuai and Tang, Ruiming and He, Xiuqiang and Hao, Jianye and Yu, Yong},
  booktitle={Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={1259--1268},
  year={2021}
}
```
