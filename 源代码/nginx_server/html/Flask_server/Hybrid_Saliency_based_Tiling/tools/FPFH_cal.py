import numpy as np 

def compute_normal(p_j, neighbors):
    mask = np.all(neighbors != p_j, axis=1)
    filtered_neighbors = neighbors[mask]
    if filtered_neighbors.shape[0] < 2:  
        return np.zeros(3)
    neighbors_diff = filtered_neighbors - p_j
    cov_matrix = np.cov(neighbors_diff.T)
    cov_matrix = np.real(cov_matrix)
    if not np.isfinite(cov_matrix).all():  
        raise ValueError("ERROR!")
    eigvals, eigvecs = np.linalg.eig(cov_matrix)
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)
    normal_vector = eigvecs[:, np.argmin(eigvals)]  
    return normal_vector

def calculate_angles(p_i, n_i, p_j, n_j):
    v = np.cross(p_j - p_i, n_i)
    alpha = np.dot(v, n_j)
    alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
    phi = np.dot(n_i, p_j - p_i) / np.linalg.norm(p_j - p_i)
    w = np.cross(n_i, v)  
    theta = np.arctan2(np.dot(w, n_j), np.dot(n_i, n_j))  
    theta = (theta + np.pi) % (2 * np.pi) - np.pi
    return alpha, phi, theta


def quantize_angle(angle, b):
    return int(np.floor((angle + np.pi) / (2 * np.pi) * b))


def compute_fpfh(i, points, R=5, b=11):
    p_i = points[i]
    neighbors = points[i+1:R+i+1]
    n_i = compute_normal(p_i, neighbors)
    triads = []
    for p_j in neighbors:
        rows_to_delete = np.where(np.all(np.isclose(neighbors, p_j), axis=1))[0]
        filtered_neighbors = np.delete(neighbors, rows_to_delete, axis=0)
        n_j = compute_normal(p_j,  filtered_neighbors)
        alpha, phi, theta = calculate_angles(p_i, n_i, p_j, n_j)
        triads.append((alpha, phi, theta))
    histogram = np.zeros((3, b))
    for alpha, phi, theta in triads:
        alpha_bin = quantize_angle(alpha, b)
        phi_bin = quantize_angle(phi, b)
        theta_bin = quantize_angle(theta, b)
        histogram[0, alpha_bin] += 1
        histogram[1, phi_bin] += 1
        histogram[2, theta_bin] += 1
    SPFH = histogram.flatten()
    FPFH = SPFH.copy()
    for p_j in neighbors:
        neighbor_histogram = np.zeros((3, b)) 
        neighbor_SPFH = neighbor_histogram.flatten()
        distance = np.linalg.norm(p_j - p_i)
        FPFH += neighbor_SPFH / distance
    return FPFH















