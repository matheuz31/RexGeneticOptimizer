import torch
import random
import math
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from neural_network import Transition

EPS_START = 0.8  # Iniciar com uma taxa de exploração um pouco mais baixa
EPS_END = 0.1   # Terminar com uma taxa de exploração um pouco mais alta
EPS_DECAY = 300 # Diminuir a taxa de exploração mais lentamente
steps_done = 0

def select_action(state, n_actions, policy_net, device):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    if sample > eps_threshold:
        with torch.no_grad():
            # A rede neural seleciona a ação com base no estado
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        # Escolhe uma ação aleatoriamente
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

BATCH_SIZE = 128
GAMMA = 0.999  # Fator de desconto

def optimize_model(policy_net, target_net, optimizer, memory, device):
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Converte o batch de Transitions para um batch de estados, ações, etc.
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Calcula Q(s_t, a) - os Q valores para as ações que foram tomadas
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Calcula V(s_{t+1}) para todos os estados seguintes
    next_state_values = torch.zeros(BATCH_SIZE)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # Calcula os Q valores esperados
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Computa a perda Huber
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    print(f"Perda na última otimização: {loss.item()}")

    # Otimiza o modelo
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    