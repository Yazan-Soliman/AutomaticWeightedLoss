# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class AutomaticWeightedLoss(nn.Module):
    """
    Automatically weighted multi-task loss.

    Inspired by:
        - Auxiliary tasks in multi-task learning (Liebel et al., 2018).
          https://arxiv.org/abs/1805.06334
        - Multi-task loss using homoscedastic uncertainty for weighting.
          https://arxiv.org/abs/1705.07115
        - Forked and extended from git@github.com:Yazan-Soliman/AutomaticWeightedLoss.git 

    Args:
        num (int): Number of tasks.
        init_weights (list, optional): Initial weights for each task. Defaults to 1 for all tasks.
        reg_type (str): Type of regularization ('log', 'l1', or 'l2').
        reg_strength (float): Strength of the regularization term.
        
    Example:
        >>> import torch
        >>> loss1 = torch.tensor(1.0, requires_grad=True)
        >>> loss2 = torch.tensor(2.0, requires_grad=True)
        >>> awl = AutomaticWeightedLoss(2, reg_type='l1', reg_strength=0.1)
        >>> combined_loss = awl(loss1, loss2)
        >>> combined_loss.backward()
        >>> print("Learned weights:", awl.params)
    """
    def __init__(self, num=2, init_weights=None, reg_type='log', reg_strength=1.0):
        super(AutomaticWeightedLoss, self).__init__()
        if init_weights is not None and len(init_weights) != num:
            raise ValueError("The length of init_weights must match the number of tasks (num).")
        if reg_type not in ['log', 'l1', 'l2']:
            raise ValueError("Unsupported reg_type. Choose from 'log', 'l1', or 'l2'.")

        if init_weights is None:
            params = torch.ones(num, requires_grad=True)
        else:
            params = torch.tensor(init_weights, requires_grad=True, dtype=torch.float32)
        
        self.params = nn.Parameter(params)
        self.reg_type = reg_type
        self.reg_strength = reg_strength

    def forward(self, *losses):
        if len(losses) != len(self.params):
            raise ValueError(f"Number of losses ({len(losses)}) must match the number of tasks ({len(self.params)}).")

        loss_sum = 0
        for i, loss in enumerate(losses):
            # Weighted loss
            weighted_loss = 0.5 / (self.params[i] ** 2) * loss
            
            # Regularization term
            if self.reg_type == 'log':
                regularization = self.reg_strength * torch.log(1 + self.params[i] ** 2)
            elif self.reg_type == 'l1':
                regularization = self.reg_strength * torch.abs(self.params[i])
            elif self.reg_type == 'l2':
                regularization = self.reg_strength * (self.params[i] ** 2)
            # Accumulate loss
            loss_sum += weighted_loss + regularization

        return loss_sum


if __name__ == '__main__':
    awl = AutomaticWeightedLoss(2)
    print(awl.parameters())
