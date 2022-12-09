# Exp
def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))  # y label 범위 (-10 ~ 10) 고정 자연상수를 씌운 후 반환

# Softplus
def softplus_evidence(y):
    return F.softplus(y)  # y 레이블에 대한 softplus 함수를 거친 값 반환

def relu_evidence(y):
    return F.relu(y)  # 레이블에 대한 relu 함수를 거친 값 반환

# KL Divergence
def kl_divergence(alpha, num_classes, device=None):
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)  # 클래스 차원을 가지고 있는 배열(모두 1 값으로 초기화 되어 있다.)

    # 알파값의 합을 구하기 위한 과정, dim = 1일 경우 row 끼리 연산, keepdim = True 일 경우 인풋값의 차원과 출력값의 차원을 동일하게 유지
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)

    # kl_divergence 은 KL(D [first_term] || D [second_term]) 으로 이루어져 있다.

    # D [first_term]
    # 6페이지의 KLD 수식을 확인
    first_term = (
        # lgamma : 감마(x)의 절대값의 자연 로그
            torch.lgamma(sum_alpha)  # log(r(Sigma_ak)) 분자 부분에 해당한다.
            - torch.lgamma(alpha).sum(dim=1, keepdim=True)  # log(r(a))
            + torch.lgamma(ones).sum(dim=1, keepdim=True)
            - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    # 두번째 텀
    second_term = (
        (alpha - ones)  # ak - 1     - A
            # digamma(ak) - digamma(Sigma(ak))     - B
            .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
            # 두 식 A, B 를 matmul 연산한 후에,
            .sum(dim=1, keepdim=True)  # matmul 연산 결과에 Sigma 를 씌운다.
    )
    # KLD 의 최종 값 (논문에서는 유니폼된 KLD 를 구했다.)
    kl = first_term + second_term  # 첫번째 텀과 두번째 텀의 합을 구하면 최종 결과를 구할 수 있다.
    return kl


# L(theta) 반환 함수
def loglikelihood_loss(y, alpha, device=None):
    y = y.to(device)  # y의 값을 cuda 환경으로 올린다.
    alpha = alpha.to(device)  # 알파 값도 cuda 환경으로 올린다.
    S = torch.sum(alpha, dim=1, keepdim=True)  # 알파값의 합을 S

    # L_err term 의 부분, row 끼리 연산, 입력 차원에 대해서 출력 차원을 유지한다.
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    # L_var term 의 부분, row 끼리 연산, 입력 차원에 대해서 출력 차원을 유지한다.
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    # 위에서 구한 두 수식을 합하면 최종 결과 값을 구할 수 있다
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood


# 평균제곱오차
def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, device=None):
    y = y.to(device)  # y의 값을 cuda 환경으로 올린다.
    alpha = alpha.to(device)  # 알파 값도 cuda 환경으로 올린다.

    # 바로 위 함수에서 정의한 L(theta)값을 구한 후, cuda 환경으로 올린다.
    loglikelihood = loglikelihood_loss(y, alpha, device=device)

    # min(1.0, t/10), 논문에서는 epoch_num = t, annealing_step = 10 으로 지정
    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32)
    )

    # hat_a_i 의 값을 구하는 것, 이 값으로 KLD 를 구하자
    kl_alpha = (alpha - 1) * (1 - y) + 1

    # uniformed KLD 값과 annealing_coef 를 곱한 후, 위에서 정의한
    # loglikelihood 와 더하면 최종적인 연산 값을 구할 수 있다.
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return loglikelihood + kl_div


# Evidential Deep Learning Loss
def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device=None):
    y = y.to(device)  # y의 값을 cuda 환경으로 올린다.
    alpha = alpha.to(device)  # 알파 값도 cuda 환경으로 올린다.
    S = torch.sum(alpha, dim=1, keepdim=True)  # 알파의 합을 S로 정의

    # 4페이지의 3번식 부분
    # Sigma[y * (log(S) - log(a))]
    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    # min(1.0, t/10), 논문에서는 epoch_num = t, annealing_step = 10 으로 지정
    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32)
    )

    # hat_a_i 의 값을 구하는 것, 이 값으로 KLD 를 구하자
    kl_alpha = (alpha - 1) * (1 - y) + 1

    # uniformed KLD 값과 annealing_coef 를 곱한 후, 위에서 정의한
    # A 와 더하면 최종적인 연산 값을 구할 수 있다.
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div


def edl_mse_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    # output을 넣어 evidence를 구한 후,
    evidence = relu_evidence(output)
    # 논문에서 a = e + 1 을 가진 디리클레를 정의하였으으니, 아래와 같다.
    alpha = evidence + 1
    # 위를 적용한 후, mse_loss 를 구한다.
    loss = torch.mean(
        mse_loss(target, alpha, epoch_num, num_classes, annealing_step, device=device)

    )
    return loss


"""
edl_log_loss 와 edl_digamma_loss 의 차이는
function 을 log 를 사용할 것인지, digamma 를 사용할 것인지의 차이이다.
"""


def edl_log_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    # output을 넣어 evidence를 구한 후,
    evidence = relu_evidence(output)
    # 논문에서 a = e + 1 을 가진 디리클레를 정의하였으으니, 아래와 같다.
    alpha = evidence + 1
    # 위를 적용한 후, edl_loss 를 구한다.
    loss = torch.mean(
        edl_loss(
            torch.log, target, alpha, epoch_num, num_classes, annealing_step, device
        )
    )
    return loss


def edl_digamma_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    # output을 넣어 evidence를 구한 후,
    evidence = relu_evidence(output)
    # 논문에서 a = e + 1 을 가진 디리클레를 정의하였으니, 아래와 같다.
    alpha = evidence + 1
    # 위를 적용한 후, edl_loss 를 구한다.
    loss = torch.mean(
        edl_loss(
            torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, device
        )
    )
    return loss
