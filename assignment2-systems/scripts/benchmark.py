from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

import timeit
import torch


torch.set_default_device('cuda')

model = BasicsTransformerLM(
    vocab_size=256,
    context_length=100,
    d_model=128,
    num_layers=5,
    num_heads=8,
    d_ff=512,
    rope_theta=1e-5
)
optimizer = AdamW(
    model.parameters()
)

data = torch.randint(
    low=0,
    high=256,
    size=(8, 100)
)
labels =  torch.randint(
    low=0,
    high=256,
    size=(8, 100)
)


fw_times = []
bw_times = []


for _ in range(6):
    optimizer.zero_grad()

    torch.cuda.synchronize()
    start_time = timeit.default_timer()
    output = model(data)
    torch.cuda.synchronize()
    fw_time = timeit.default_timer() - start_time
    fw_times.append(fw_time)

    loss = cross_entropy(output, labels)
    torch.cuda.synchronize()
    start_time = timeit.default_timer()
    loss.backward()
    torch.cuda.synchronize()
    bw_time = timeit.default_timer() - start_time
    bw_times.append(bw_time)

print("avg_fw_time:\t", sum(fw_times[-5:]) / len(fw_times[-5:]) )
print("avg_bw_time:\t", sum(bw_times[-5:]) / len(bw_times[-5:]) )
