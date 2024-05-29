# Robust Value Entropy Search

This is companion code for the paper "Robust Entropy Search for Safe Efficient Bayesian Optimization" by D. Weichert et al., UAI 2024. 

## Installation guide

In the root directory of the repository execute the following commands:

```shell
conda env create --file=environment.yml
conda activate res
```

## Execution

To reproduce the experimental results, please run 

```shell
python main.py
```

This automatically creates subdirectories and files in the `Results/` directory for each algorithm and experiment.

## Analysis

To analyse the results and create the figures presented in the paper, please run the Jupyter Notebook analyse_synthetic_experiments.ipynb .

## License
Copyright (c) 2024 Fraunhofer IAIS

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Author: Dorina Weichert, dorina.weichert@iais.fraunhofer.de