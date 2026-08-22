import flwr as fl

def get_strategy():
    return fl.server.strategy.FedAvg(
        fraction_fit=1.0,        
        fraction_eval=1.0,
        min_fit_clients=3,       
        min_eval_clients=3,
        min_available_clients=3,
        on_fit_config_fn=lambda rnd: {"epochs": 1, "batch_size": 32},
        on_evaluate_config_fn=lambda rnd: {"val_steps": 10},
    )
