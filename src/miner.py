# Imports
import time
import shared
import argparse
import bittensor as bt

# --- Build config.
parser = argparse.ArgumentParser()
parser.add_argument( '--device', type = str, default='cuda' )
bt.wallet.add_args( parser )
bt.axon.add_args( parser )
bt.logging.add_args( parser )
config = bt.config( parser )
bt.logging( config = config )

# --- Hparams + Tokenizer
bs = 8
sq = 512

# --- Build objects.
wallet = bt.wallet( config = config ).create_if_non_existent()
model = shared.get_model().to( config.device )
dataloader = shared.get_dataloader( bs, sq )

# --- Define forward function.
# Accepts the model state, loads the state into the model and computes a set of 
# aggregated gradients. The gradients are then packed into the response and returned.
def compute_gradients( synapse: shared.ComputeGradients ) -> shared.ComputeGradients:
    bt.logging.info(f'Start forward')
    # Clear previous gradients.
    model.zero_grad()
    model.load_state_dict( synapse.deserialize_state() )

    # Apply n_steps gradients aggregations.
    step = 0
    while True:
        batch = next( dataloader )
        inputs = batch.to( config.device )            
        outputs = model( inputs, labels = inputs )
        loss = outputs.loss/synapse.n_steps      
        loss.backward()
        bt.logging.info(f'Step: {step}, Loss: {loss.item() * synapse.n_steps}')
        if step >= synapse.n_steps: break
        else: step += 1

    # Serialize the gradients onto the model state.
    synapse.serialize_state( state_dict = { k: v.grad for k, v in model.named_parameters() } ) 
    synapse.loss = loss.item()
    bt.logging.info(f'End forward')
    return synapse

# TODO (const): use the metagraph.
# Turn off verify requests since the validator does not have hotkey information.
# only the ports are passed. 
def verify( synapse: shared.ComputeGradients ) -> None: pass

# --- Run server.
# Builds the axon from the wallet.
axon = bt.axon( wallet = wallet, config = config )
axon.attach( compute_gradients, verify_fn = verify)
axon.start()
bt.logging.success(f'Starting miner: {axon}')
while True: 
    time.sleep( 1 )

