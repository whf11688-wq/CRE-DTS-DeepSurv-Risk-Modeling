import torch
import torch.nn as nn
import torch.optim as optim


class DeepSurvLoss(nn.Module):
    """
    Custom Negative Log Partial Likelihood Loss Function for CRE-DTS (DeepSurv).
    This computes risk sets dynamically based on time-to-event data.
    """

    def __init__(self, alpha=1e-4):
        super(DeepSurvLoss, self).__init__()
        self.alpha = alpha  # L2 Regularization hyperparameter

    def forward(self, risk_pred, times, events, model_params):
        """
        risk_pred: Tensor of shape (N, 1) -> h_theta(X) from Neural Network
        times: Tensor of shape (N,) -> Time-to-event T_i
        events: Tensor of shape (N,) -> Censoring indicator delta_i (1=Default, 0=Censored)
        model_params: PyTorch model parameters for L2 regularization
        """
        # Sort data by survival times in descending order to construct dynamic Risk Sets R(T_i)
        sorted_times, indices = torch.sort(times, descending=True)
        sorted_risk = risk_pred[indices].squeeze()
        sorted_events = events[indices].squeeze()

        # Compute log-sum-exp of risk for the risk sets dynamically
        # exp(h_j) accumulation for j present in R(T_i)
        exp_risk = torch.exp(sorted_risk)
        cum_exp_risk = torch.cumsum(exp_risk, dim=0)
        log_cum_exp_risk = torch.log(cum_exp_risk)

        # Calculate partial likelihood only for observed default events (delta_i = 1)
        uncensored_likelihood = sorted_risk - log_cum_exp_risk
        event_likelihood = uncensored_likelihood * sorted_events

        # Average negative log likelihood
        num_events = torch.sum(sorted_events)
        if num_events == 0:
            loss = -torch.sum(event_likelihood)
        else:
            loss = -torch.sum(event_likelihood) / num_events

        # Add L2 Regularization (Weight Decay)
        l2_reg = sum(torch.sum(param ** 2) for param in model_params)
        total_loss = loss + self.alpha * l2_reg

        return total_loss


class CREDTSNetwork(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for Non-linear CRE Risk Factor Extraction.
    """

    def __init__(self, input_dim):
        super(CREDTSNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Linear(16, 1)  # Output node generates the continuous log-hazard ratio h_theta(X)
        )

    def forward(self, x):
        return self.network(x)


# --- Simulation and Execution Workflow Example ---
if __name__ == "__main__":
    # Example parameterization for Metropolitan Plaza scenario
    # Input Dim = 12 (Traditional Financials + NLP Litigation Features + IoT Power Grid Data)
    batch_size = 100
    feature_dim = 12

    # Generate dummy tensors representing simulated property data matrices
    simulated_features = torch.randn(batch_size, feature_dim)
    simulated_times = torch.randint(1, 60, (batch_size,)).float()  # Timeline: 1 to 60 months
    simulated_events = torch.randint(0, 2, (batch_size,)).float()  # 1=Default, 0=Censored

    # Instantiate Model, Loss, and Optimizer
    model = CREDTSNetwork(input_dim=feature_dim)
    criterion = DeepSurvLoss(alpha=1e-4)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # Forward Pass and Loss Computation
    model.train()
    optimizer.zero_grad()

    risk_predictions = model(simulated_features)
    loss = criterion(risk_predictions, simulated_times, simulated_events, model.parameters())

    # Backward Pass (Gradient Optimization)
    loss.backward()
    optimizer.step()

    print(f"CRE-DTS Model Execution Successful. Initial Training Loss: {loss.item():.4f}")