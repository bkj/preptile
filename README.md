### preptile

pytorch reptile

#### Installation

```
conda create -n reptile_env python=3.6 pip -y
source activate reptile_env

pip install -r requirements.txt
conda install pytorch torchvision cuda90 -c pytorch -y
pip install git+https://github.com/bkj/basenet
```

#### Usage

See `run.sh` for usage.

#### To DO

- Test on MiniImagenet
- Non-transductive evaluation
- Are the batchnorm parameters correct?
- Test w/ rotation
- Test