import numpy as np
import torch

def compare_weights(weights1, weights2):
    w1 = weights1
    w2 = weights2
    names1 = []
    names2 = []
    if(len(w1) != len(w2)):
        return False
    
    for x in (w1):
        names1.append(x)
    for y in (w2):
        names2.append(y)
    if(len(w1) == len(names1)):
        count = 0
        for j in range(len(names1)):
            if(np.array_equal(w1[names1[j]], w2[names2[j]])):
                count += 1
                continue
            else:
                return False
    
        print("Total dict:", count)
        return True

@torch.no_grad()   
def RMSprop_update(params,
                    grads,
                    square_avgs,
                    weight_decay,
                    lr,
                    eps,
                    alpha):

    for i, param in enumerate(params):
        grad = torch.Tensor(grads[i])
        square_avg = square_avgs[i]

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
        avg = square_avg.sqrt().add_(eps)
        param.addcdiv_(grad, avg, value=-lr)
    return params
