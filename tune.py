import train
import optuna

def objective( trial ):
    x = trial.suggest_float('x', -10, 10)
    y = trial.suggest_float('y', -10, 10)
    max_batches = 300
    batches_per_eval = 100
    batches_per_join = 100
    num_peers = 4
    join_prob = trial.suggest_float('join_prob', 0.00001, 1)
    lr = trial.suggest_float('lr', 0.00001, 0.01)
    bs = int( trial.suggest_float('bs', 16, 256) )
    return train.run( 
        max_batches = max_batches,
        batches_per_eval = batches_per_eval,
        num_peers = num_peers,
        join_prob = join_prob,
        batches_per_join = batches_per_join,
        lr = lr,
        bs = bs,
    )
study = optuna.create_study()
study.optimize( objective, n_trials = 10 )
print( 'best_params', study.best_params )  # E.g. {'x': 2.002108042}
