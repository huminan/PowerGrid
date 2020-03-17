# PowerGrid
## Introduction

Power grid simulator, in order to implement centralized and distributed [[1]](https://www.sciencedirect.com/science/article/pii/S0005109814004580) least square state estimation and bad data detection. Also, simplest False data injection Attack.

This project up to now have built a DC/AC power grid simulator environment with IEEE 118-bus system (other systems will be added to this project for soon once I find relavent documents), it can

- Least square state estimation
- Distributed least square state estimation
- Simplest FDI attack


## Require

`pip install -r requirements.txt`

## Framework need to do

- [ ] IEEE $< 118$ bus and 300 bus system

- [ ] Classical kalman filter state estimation (Centralized and Distributed)

- [ ] Communication delay (how about introduce ns3?)

- [ ] A better GUI (FDI configuration aquired)

- [ ] Make the algorithm more suitable for distribution 


## Alogrithm need to do

- [ ] Other classical distributed state estimation algorithms
- [ ] Asychronism type of these distributed algorithms
- [ ] FDI Attack with implement topology information [[2]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.714.756&rep=rep1&type=pdf)
- [ ] FDI Defend [[5]](https://ptolemy.berkeley.edu/projects/truststc/conferences/10/CPSWeek/papers/scs1_paper_2.pdf)
- [ ] FDI Detection by sparse optimization [[6]](https://sci-hub.se/https://ieeexplore.ieee.org/abstract/document/6740901/), machine learning [[7]](https://sci-hub.se/https://ieeexplore.ieee.org/abstract/document/7063894/)

## Have done

- [x] Dynamic (time varying state space equation) power system
- [x] AC measure model

- [x] WSNs model

- [x] A crude GUI

- [x] Richardson based distributed algorithm ((A)sychronism)
- [x] Stochastic based distributed algorithm

- [x] FDI Attack with only measurement information (ICA, PCA) [[3]](https://sci-hub.se/https://ieeexplore.ieee.org/abstract/document/6102326/) [[4]](https://sci-hub.se/https://ieeexplore.ieee.org/abstract/document/7001709/)
