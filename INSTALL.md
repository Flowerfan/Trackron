## Installation

### Requirements
- Python ≥ 3.6
- PyTorch ≥ 1.7 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
- OpenCV 



### Env Setting
For pytorch=1.10.2 and cuda11.3
```
conda env create -f env.yaml
or
pip install -r req.txt
``` 
then build the env

```
# should export cuda(11.3) to $PATH and $LD_LIBRARY first
python setup.py install/develop
or
pip install -v -e .
```

