## Federated GPT2

Miners accept requests from the validator containing a model state dict and return a gradient over that model with respect to a dataset. The gradients are aggregated on the validator and applied to the model. We use a GPT2 model architecture and the Falcon Refined Web dataset. The optimizer exists only on the validator side. Miner are free to compute gradients in any manner. 

TODO: introduce a ranking method. 

## Install

```bash
python -m pip install -r requirements.txt
```

## Running

First run your miners
```bash
# Miner script.
#   python src/miner.py
#
# Miner wallet name
#    --wallet.name miners Miner wallet name
#
# Miner hotkey must be distinct per miner
#    --wallet.hotkey M1
#
# Select a device (different for each miner), if you dont have a GPU pass 'cpu'  
#    --device cuda:1  
#
# Each miner must have a separate port here (also different for each miner)
#    --axon.port 8091 

# Run first miner
python src/miner.py --wallet.name miners --wallet.hotkey M1 --device cuda:1 --axon.port 8091

# Run your second miner
python src/miner.py --wallet.name miners --wallet.hotkey M2 --device cuda:2 --axon.port 8092 
```

Second run your validator/trainer on the same machine.
```bash
#
# Script name:
#   python src/validator.py
#
# The validator wallet name:
#    --wallet.name validator 
#
# The validator hotkey name:
#    --wallet.hotkey default 
#
# The validator device, different that miners:
#    --device cuda:0 
#
# Enter the ports for each of the miners you are running:
#    --axons 8091 8092 

# Run the validator
python src/validator.py --wallet.name validator --wallet.hotkey default --device cuda:0 --axons 8091 8092
```

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
