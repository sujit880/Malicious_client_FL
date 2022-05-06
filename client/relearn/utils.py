def RMSprop_update(params,
                    grads,
                    square_avgs,
                    weight_decay,
                    lr,
                    eps,
                    alpha):
    """Functional API that performs rmsprop algorithm computation.
    See :class:`~torch.optim.RMSProp` for details.
    """

    for i, param in enumerate(params):
        grad = grads[i]
        square_avg = square_avgs[i]

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
        avg = square_avg.sqrt().add_(eps)
        param.addcdiv_(grad, avg, value=-lr)
    return params
