import pickle
import time
import torch

# Required import from sbi.
from sbi.inference import MNLE
from sbi.utils import BoxUniform


N = 100000
training_batch_size = 100

with open("data/ddm_training_data.p", "rb") as fh:
    theta, x1d, tho, xos = pickle.load(fh).values()
    # choices are encoded in the sign of the reaction times.
    # decode and put them in separate columns.
    x = torch.zeros(x1d.shape[0], 2)
    x[:, 0] = abs(x1d.squeeze())
    x[x1d.squeeze() > 0, 1] = 1

theta = theta[:N]
x = x[:N]

prior = BoxUniform(torch.tensor([-2., 0.5, 0.3, 0.2]), 
                   torch.tensor([2., 2.0, 1.7, 1.8]))

tic = time.time()
trainer = MNLE(prior=prior)
trainer = trainer.append_simulations(theta, x)
mnle = trainer.train(training_batch_size=training_batch_size, 
                     show_train_summary=True)
print("MNLE took", time.time() - tic, "seconds")
