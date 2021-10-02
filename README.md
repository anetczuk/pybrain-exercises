# Exercises with *pybrain*

Exercises with *pybrain*.


## Pybrain running limitations

Pybrain seems to be not maintained anymore and for this reason is not compatible with modern python language. It has been found, that library works with following dependencies:
- *Python* language version 2.7
- *scipy* version 0.19.1
- *numpy* version 1.8.2
- *matplotlib* version 2.2.5
- *pyode* version 1.2.1 with *libode6* version 2:0.14-2 on system (might work with newer system backends)
- *pyopengl* version 3.1.5 with *libopengl0* version 1.0.0-2 on system (might work with newer system backends)
- *pillow* version 6.2.2

Moreover, because of python's `PIL` changes, *pybrain* does not work (environments viewing) straight form project's GitHub repository. For all those reasons script `lib/pybrain/install.sh` is introduced to facilitate installation of the library.


## Virtual environment

For all reasons pointed above user is encouraged to use virtual environment. Configuration of the environment can be as simple as execution of `tools/installvenv.sh` script. Script assumes that following components are preinstalled:
- *Python* version 2
- *virtualenv* module (tested on version 20.8.1)

Script alongside the environemnt installs proper *pybrain* and all it's dependencies stated in previous paragraph.

Starting environemt after installation can be done by execution of script `tools/startvenv.sh`. 


## Reinforcement learning environments preview

[![shipsteer](doc/env/shipsteer-small.png "shipsteer")](doc/env/shipsteer-big.png)
[![acrobot](doc/env/acrobot-small.png "acrobot")](doc/env/acrobot-big.png)
[![ccrl-glass](doc/env/ccrl-glass-small.png "ccrl-glass")](doc/env/ccrl-glass-big.png)
[![flexcube](doc/env/flexcube-small.png "flexcube")](doc/env/flexcube-big.png)
[![johnnie](doc/env/johnnie-small.png "johnnie")](doc/env/johnnie-big.png)


## Alternative machine learning libraries

- scikit-learn (https://scikit-learn.org) 
- Apache's MLlib (https://spark.apache.org/mllib)


## References

- pybrain (http://pybrain.org/)
