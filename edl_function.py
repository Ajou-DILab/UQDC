# Exp
def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))  # Returns the result of applying the natural exponent to y, clamped to the range (-10 to 10)

# Softplus
def softplus_evidence(y):
    return F.softplus(y)  # Returns the result of applying the softplus function to y

def relu_evidence(y):
    return F.relu(y)  # Returns the result of applying the ReLU function to y

# KL Divergence
def kl_divergence(alpha, num_classes, device=None):
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)  # An array initialized with ones for the class dimension

    # Process to calculate the sum of alpha values; for dim=1, rows are summed, and keepdim=True retains input-output dimensions
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)

    # kl_divergence is composed of KL(D [first_term] || D [second_term])

    # D [first_term]
    # Refer to the KLD formula on page 6
    first_term = (
        # lgamma: Natural logarithm of the absolute value of gamma(x)
            torch.lgamma(sum_alpha)  # Corresponds to the numerator part of log(r(Sigma_ak))
            - torch.lgamma(alpha).sum(dim=1, keepdim=True)  # log(r(a))
            + torch.lgamma(ones).sum(dim=1, keepdim=True)
            - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    # Second term
    second_term = (
        (alpha - ones)  # ak - 1     - A
            # digamma(ak) - digamma(Sigma(ak))     - B
            .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
            # Perform a matmul operation for A and B,
            .sum(dim=1, keepdim=True)  # Sum the result of matmul operation
    )
    # Final value of KLD (uniform KLD calculated in the paper)
    kl = first_term + second_term  # Final result is obtained by summing the first and second terms
    return kl


# Function to return L(theta)
def loglikelihood_loss(y, alpha, device=None):
    y = y.to(device)  # Transfer y to CUDA environment
    alpha = alpha.to(device)  # Transfer alpha to CUDA environment
    S = torch.sum(alpha, dim=1, keepdim=True)  # Sum of alpha values as S

    # Part of L_err term, performs row-wise operations and retains input-output dimensions
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    # Part of L_var term, performs row-wise operations and retains input-output dimensions
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    # Final result is obtained by summing the two expressions above
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood


# Mean Squared Error
def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, device=None):
    y = y.to(device)  # Transfer y to CUDA environment
    alpha = alpha.to(device)  # Transfer alpha to CUDA environment

    # Calculate L(theta) as defined in the function above and transfer it to CUDA
    loglikelihood = loglikelihood_loss(y, alpha, device=device)

    # min(1.0, t/10); in the paper, epoch_num = t, annealing_step = 10
    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32)
    )

    # Calculate hat_a_i value to compute KLD
    kl_alpha = (alpha - 1) * (1 - y) + 1

    # Multiply uniformed KLD value with annealing_coef and add it to loglikelihood to get the final result
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return loglikelihood + kl_div


# Evidential Deep Learning Loss
def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device=None):
    y = y.to(device)  # Transfer y to CUDA environment
    alpha = alpha.to(device)  # Transfer alpha to CUDA environment
    S = torch.sum(alpha, dim=1, keepdim=True)  # Define the sum of alpha as S

    # Part of equation 3 on page 4
    # Sigma[y * (log(S) - log(a))]
    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    # min(1.0, t/10); in the paper, epoch_num = t, annealing_step = 10
    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32)
    )

    # Calculate hat_a_i value to compute KLD
    kl_alpha = (alpha - 1) * (1 - y) + 1

    # Multiply uniformed KLD value with annealing_coef and add it to A to get the final result
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div


def edl_mse_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    # Calculate evidence by applying the function to the output
    evidence = relu_evidence(output)
    # In the paper, Dirichlet is defined as a = e + 1, hence:
    alpha = evidence + 1
    # Apply this and calculate mse_loss
    loss = torch.mean(
        mse_loss(target, alpha, epoch_num, num_classes, annealing_step, device=device)

    )
    return loss


"""
The difference between edl_log_loss and edl_digamma_loss lies in whether
the log function or the digamma function is used.
"""


def edl_log_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    # Calculate evidence by applying the function to the output
    evidence = relu_evidence(output)
    # In the paper, Dirichlet is defined as a = e + 1, hence:
    alpha = evidence + 1
    # Apply this and calculate edl_loss
    loss = torch.mean(
        edl_loss(
            torch.log, target, alpha, epoch_num, num_classes, annealing_step, device
        )
    )
    return loss


def edl_digamma_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    # Calculate evidence by applying the function to the output
    evidence = relu_evidence(output)
    # In the paper, Dirichlet is defined as a = e + 1, hence:
    alpha = evidence + 1
    # Apply this and calculate edl_loss
    loss = torch.mean(
        edl_loss(
            torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, device
        )
    )
    return loss
