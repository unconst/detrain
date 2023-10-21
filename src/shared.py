# --- Imports
import torch
import typing
import random
import requests
import bittensor as bt
from torch.utils.data import DataLoader, IterableDataset
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer

# --- Wire Protocol.
class ComputeGradients( bt.Synapse ):

    n_steps: int = 10
    loss: float = None
    
    state_dict: typing.Optional[ typing.Dict[ str, bt.Tensor ] ] = None

    def serialize_state( self, state_dict: typing.Dict[str, torch.Tensor ] ):
        self.state_dict = { key: bt.Tensor.serialize( value ) for key, value in state_dict.items() }

    def deserialize_state( self ) -> typing.Dict[str, torch.Tensor ]:
        return { key: value.deserialize() for key, value in self.state_dict.items() }

# --- Model Arch
def get_model():
    return GPT2LMHeadModel( GPT2Config(
        n_layer = 12, 
        n_head = 12,
    ))

# --- Dataloader
# Using the Falcon refined web huggingface api directly. We cycle through all 968000015 rows at random.
# The iterator of this class returns sequences of length `sequence_length` which are randomly pulled from
# the Falcon refined web corpus. 
class FalconDataset(IterableDataset):
    def __init__(self, tokenizer, sequence_length):
        self.buffer = []
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.num_rows_total = 968000015
        self.num_rows_per_page = 100
        self.base_url = "https://datasets-server.huggingface.co/rows"
        self.params = {
            "dataset": "tiiuae/falcon-refinedweb",
            "config": "default",
            "split": "train"
        }
        
    def get_random_data(self):
        # Gets a random page of data from the URL and returns the json rows.
        random_offset = random.randint(0, self.num_rows_total - self.num_rows_per_page)
        self.params["offset"] = random_offset
        self.params["limit"] = self.num_rows_per_page
        response = requests.get(self.base_url, params=self.params)
        if response.status_code == 200:
            return response.json()["rows"]
        else:
            return []

    def __iter__(self):
        while True:
            # If buffer is empty, fetch a new random data page
            if not self.buffer:
                data = self.get_random_data()
                for row in data:
                    content = row["row"]["content"]
                    self.buffer += self.tokenizer(content)["input_ids"]
                    self.buffer += [self.tokenizer.eos_token_id]
            # Yield data in sequence length chunks until buffer is exhausted
            if len(self.buffer) > self.sequence_length:
                yield torch.tensor(self.buffer[:self.sequence_length])
                self.buffer = self.buffer[self.sequence_length:]
            else:
                # If there's not enough data in the buffer for another sequence,
                # we will just clear the buffer and fetch new data in the next iteration.
                self.buffer = []


# --- Get dataloader.
def get_dataloader( bs, sq ):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    dataset = FalconDataset( tokenizer = tokenizer, sequence_length = sq )
    dataloader = iter( DataLoader( dataset, batch_size = bs, num_workers = 1 ) )
    return dataloader