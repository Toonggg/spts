### Python 3 port of spts 
In this repository, [spts](https://github.com/mhantke/spts) is being developed in Python 3 after
migrating Python 2 specific code to Python 3. The denoising module can now be compiled under Python 3. 
The Python 3 port of spts is compatible with Python versions 3.7.x, 3.8.x, and 3.9.x. 

## Installing spts
To install spts, start by cloning this repository:
```bash
git clone https://github.com/Toonggg/spts.git
```
Before installing, make sure to check what Python version your system uses:
```bash
which python 
which python3 
```
The Python 3 port of spts only supports Python `3.7.x`, `3.8.x`, and `3.9.x`. If only Python 3 is installed in your current environment, 
install in the following way: 
```bash
python setup.py --install 
```
Otherwise, it is important to specify Python 3: 
```bash
python3 setup.py --install 
```

### Dependencies
Install additional dependencies not included in the `setup.py` script using `pip install`/`pip3 install` or conda (pip links provided in list below):
* [expiringdict](https://pypi.org/project/expiringdict/)
* [olefile](https://pypi.org/project/olefile/)
* [typing](https://pypi.org/project/typing/)

## TO-DOs 
* In the `analysis.py` function, label integration method doesn't work yet. This needs to be looked into in the future,
so that when this is used, it works... 
