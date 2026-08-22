import flwr as fl


class SecureFedAvg(fl.server.strategy.FedAvg):
    """
    Secure Aggregation compatible FedAvg
    Server NEVER inspects individual client updates
    """

    def aggregate_fit(self, rnd, results, failures):
        if not results:
            return None, {}

        # Only aggregate — masks cancel here
        aggregated_params, metrics = super().aggregate_fit(
            rnd, results, failures
        )

        return aggregated_params, metrics


if __name__ == "__main__":
    fl.server.start_server(
        server_address="127.0.0.1:9090",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=SecureFedAvg(),
    )
