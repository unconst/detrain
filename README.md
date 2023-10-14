## Federated Training on MNIST Dataset

This script, `train.py`, trains an ensemble of models on the MNIST dataset. The ensemble consists of multiple neural network models trained in parallel and independently. Periodically the models' parameters are averaged via a join operation.

### Dependencies:

- torch
- torchvision
- pandas
- itertools
- argparse
- rich

### Setup:

Ensure that you have the required dependencies installed:

```bash
python -m pip install -r requirements.txt
or
python -m pip install torch torchvision pandas rich argparse
```

### How to Run:

Execute the script from the command line:

```bash
python train.py --max_batches=10000 --batches_per_eval=1000 --num_nodes=5 --join_prob=0.1 --batches_per_join=1000
```

### Command Line Arguments:

1. `--max_batches`: The total number of mini-batches that will be processed during training. Default is `10000`.
2. `--batches_per_eval`: Specifies how frequently the models should be evaluated. Default is `1000`.
3. `--num_nodes`: The number of individual model instances or "nodes" in the ensemble. Default is `5`.
4. `--join_prob`: The probability that a pair of models will have their parameters averaged. Default is `0.1`.
5. `--batches_per_join`: Defines the frequency at which the model parameters may be averaged. Default is `1000`.

### Output:

As the models are trained, their performances will be displayed on the console. The script tracks the following metrics:

- **base**: Accuracy of the first model in the ensemble.
- **max**: Accuracy of the best model in the ensemble.
- **min**: Accuracy of the worst model in the ensemble.
- **mean**: Average accuracy of the ensemble.
- **maxwin**: Difference between the base value and the best model's accuracy.
- **minwin**: Difference between the base value and the worst model's accuracy.
- **meanwin**: Difference between the base value and the average accuracy of the ensemble.

Additionally, a CSV file (`history.csv`) is saved which logs all the above metrics for further analysis.

## Analyzing the Results: `analysis.ipynb`

After running the `train.py` script, the training metrics of the ensemble are saved in `history.csv`. To visualize and analyze these results, use the provided Jupyter notebook `analysis.ipynb`.

### Dependencies:

- pandas
- matplotlib
- jupyter

If you haven't installed these libraries, you can do so using:

```bash
pip install pandas matplotlib jupyter
```

### Model Architecture:

The neural network used for this task is a simple feed-forward network with two dense layers. The first layer has 128 neurons, and the second layer has 10 neurons corresponding to the 10 classes of the MNIST dataset.

### Notes:

- While this ensemble method provides diversity among the models, it also promotes convergence of their parameters due to the joining mechanism.
- The infinite_train_loader ensures that the training never runs out of data by cycling through the dataset.
- The joining mechanism (averaging of parameters) is based on a probability. Therefore, not all pairs of models are joined.

### License

This repository is licensed under the MIT License.

```
# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
```

---
