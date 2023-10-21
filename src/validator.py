# Imports
import time
import asyncio
import shared
import argparse
import bittensor as bt
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer, AdamW

# Build config.
parser = argparse.ArgumentParser()
parser.add_argument( '--device', type = str, default='cuda' )
parser.add_argument( '--axons', type=int, nargs='+', required=False, help="List of port numbers for local axons.")
bt.wallet.add_args( parser )
bt.axon.add_args( parser )
bt.logging.add_args( parser )
config = bt.config( parser )
bt.logging( config = config )

# --- Hparams + Tokenizer
lr = 1e-4
bs = 8
sq = 512
n_steps_per_worker = 3
n_total_steps = 1000

# Build objects.
wallet = bt.wallet( config = config ).create_if_non_existent()
model = shared.get_model().to( config.device )
model.train()
optimizer = AdamW( model.parameters(), lr=lr )

axons = []
for p in config.axons:
    axons.append( bt.axon( port = p ).info() )

# Query miner
bt.logging.success('Training...')
for gs in range( n_total_steps ):

    # Query single miner.
    synapse = shared.ComputeGradients( n_steps = n_steps_per_worker )
    synapse.serialize_state( state_dict = model.state_dict() ) 

    dendrite = bt.dendrite( wallet = wallet )
    responses = dendrite.query( axons, synapse, timeout = 100, deserialize = False )
    asyncio.get_event_loop().run_until_complete(dendrite.close_session())

    # Sum grads from all workers on master model.
    model.zero_grad()
    for resp in responses:
        remote_grads = resp.deserialize_state()
        for (name_j, param_j) in model.named_parameters():
            if name_j in remote_grads:
                if remote_grads[name_j] is not None:
                    grad_ij = remote_grads[name_j]
                    if param_j.grad is None:
                        param_j.grad = grad_ij.to( config.device )
                    else:
                        param_j.grad += grad_ij.to( config.device )
                else:
                    bt.logging.error(f'remote_grads[{name_j}] is None')
            else:
                bt.logging.error(f'{name_j} not in remote_grads:')


    # Average grads based on number of workers.
    n_workers = 1
    for (_, param_j) in model.named_parameters():
        if param_j.grad is not None:
            param_j.grad /= len( responses )

    # Step.
    optimizer.step()
    model.zero_grad()

    # Logs
    bt.logging.success( f"Step: {gs} Losses: {[ resp.loss * n_steps_per_worker if resp.loss else None for resp in responses ]}")

