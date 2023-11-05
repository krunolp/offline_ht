# Approximate Heavy Tails in Offline (Multi-Pass) Stochastic Gradient Descent

This is a repository provides source code for the [Approximate Heavy Tails in Offline (Multi-Pass) Stochastic Gradient Descent](https://arxiv.org/abs/2310.18455) paper.

## Dependencies

To install the minimal dependencies needed to use the algorithms, run in the
main directory of this repository

```commandline
pip install .
```

Alternatively, you can install the required packages using the following
command:

```commandline
pip install -r requirements.txt
```

## Usage
The repository consists of two main parts:

- ```ht_lin_reg```, which contains the linear regression experiments
- ```ht_nns```, which contains the neural network experiments
- ```figures```, which contains the code for reproducing the figures in our work



## Citing
If you refer our work in your research, please cite it as follows:

```
@misc{pavasovic2023approximate,
      title={Approximate Heavy Tails in Offline (Multi-Pass) Stochastic Gradient Descent}, 
      author={Krunoslav Lehman Pavasovic and Alain Durmus and Umut Simsekli},
      year={2023},
      eprint={2310.18455},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```




