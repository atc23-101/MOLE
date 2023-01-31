import torch 
import torch.nn.functional as F


class MOLETopk(torch.nn.Module):
    def __init__(self, model_dim, total_experts) -> None:
        super().__init__()
        self.total_experts = total_experts
        self.linear = torch.nn.Linear(model_dim, total_experts, bias=False)

    def forward(self, input):
        gates = self.linear(input)
        gate_score = F.softmax(gates, dim=-1)
        return gate_score

