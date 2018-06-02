### How to install paclages on Euler:

`mkdir -p $HOME/python/lib64/python2.7/site-packages`
`export PYTHONPATH=$HOME/python/lib64/python2.7/site-packages:$PYTHONPATH`
`module load python/2.7`

# now, install e.g. theano
`python -m pip install --install-option="--prefix=$HOME/python" theano`

see: https://people.ee.ethz.ch/~muejonat/eth-supercomputing/ for details
