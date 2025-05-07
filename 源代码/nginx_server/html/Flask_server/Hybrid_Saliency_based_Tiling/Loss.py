import numpy as np
import S_external as se


def compute_loss(N,blocks,s, inner_dispersion, exterior_rarity, inter_frame_variability, sigma=0.02):
    keys = list(blocks.keys())
    saliency_values = np.stack([inner_dispersion, exterior_rarity, inter_frame_variability], axis=0)
    L1 = 0
    epsilon = 1e-6 
    for k in range(3):
        for i in range(N):
            Sk = saliency_values[k, i]
            if Sk != 0: 
                L1 += s[i]**2 / (Sk + epsilon) + Sk * (1 - s[i])**2

    L2 = 0
    for i in range(N):
        for j in range(N):
            if i != j:
                distance = np.linalg.norm(
                    se.cal_average_points(blocks[keys[i]]['points']) - 
                    se.cal_average_points(blocks[keys[j]]['points'])
                )
                weight = np.exp(-distance**2 / (2 * sigma**2))
                L2 += weight * (s[i] - s[j])**2

    total_loss = L1 + L2
    return total_loss


def compute_gradient(N,blocks,s, inner_dispersion, exterior_rarity, inter_frame_variability, sigma=0.02):
    keys = list(blocks.keys())
    saliency_values = np.stack([inner_dispersion, exterior_rarity, inter_frame_variability], axis=0)
    epsilon = 1e-6  
    gradient = np.zeros_like(s)

    for k in range(3):
        for i in range(N):
            Sk = saliency_values[k, i]
            if Sk != 0:
                gradient[i] += 2 * s[i] / (Sk + epsilon) - 2 * (1 - s[i]) * Sk

    for i in range(N):
        for j in range(N):
            if i != j:
                distance = np.linalg.norm(
                    se.cal_average_points(blocks[keys[i]]['points']) - 
                    se.cal_average_points(blocks[keys[j]]['points'])
                )
                weight = np.exp(-distance**2 / (2 * sigma**2))
                gradient[i] += 2 * weight * (s[i] - s[j])

    return gradient


def optimize_s_with_adam(N,blocks,inner_dispersion, exterior_rarity, inter_frame_variability, sigma=0.02, lr=0.01, epochs=100):
    s = np.full(N, 0.5)
    m, v = np.zeros_like(s), np.zeros_like(s)
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8

    for epoch in range(epochs):
        grad = compute_gradient(N,blocks,s,inner_dispersion, exterior_rarity, inter_frame_variability, sigma)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** (epoch + 1))
        v_hat = v / (1 - beta2 ** (epoch + 1))
        s -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
        s = np.clip(s, 0, 1)

        if epoch % 10 == 0:
            total_loss = compute_loss(N,blocks,s, inner_dispersion, exterior_rarity, inter_frame_variability, sigma)
            print(f"Epoch {epoch}, Loss: {total_loss:.2f}")

    return s

