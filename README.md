### Python 3 port of spts 
In this repository, [spts](https://github.com/mhantke/spts) which is used to analyse 
single particle Mie scattering and written in Python 2, will be ported to Python 3. This will
involve updating older code to conform to Python 3. One of the most obvious changes is the replacement
of print statements to print() statements. 

Once the core software has been ported to Python 3, newer features could be added in the future.
Currently, the software can be installed in Python 3 (specifically Python 3.7, 3.8, and 3.9). 
The denoising extension module, written in C/C++, can now be compiled under Python 3. 
More information can be found [here](http://python3porting.com/cextensions.html). 

### Dependencies
Install additional dependencies not included in the `setup.py` script using `pip install`/`pip3 install` or conda (pip links provided in list below):
* [expiringdict](https://pypi.org/project/expiringdict/)
* [olefile](https://pypi.org/project/olefile/)
* [typing](https://pypi.org/project/typing/)

### TO-DOs 
* Add median-subtraction on raw files without providing additional background recording. 
* In the `analysis.py` function, label integration method doesn't work yet. This needs to be looked into in the future,
so that when this is used, it works... 
