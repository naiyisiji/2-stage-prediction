import torch

def welzl(P, R, device):
    if len(P) == 0 or len(R) == 3:
        return minidisk_trivial(R, device)
    P_new = P[:-1]
    p = P[-1]
    D = welzl(P_new, R, device)
    if is_in_circle(D, p):
        return D
    R_new = torch.cat((R, p.unsqueeze(0)), dim=0).to(device)
    return welzl(P_new, R_new, device)

def minidisk_trivial(P, device):
    if len(P) == 0:
        return torch.tensor([0, 0], device=device), 0
    elif len(P) == 1:
        return P[0], 0
    elif len(P) == 2:
        center = (P[0] + P[1]) / 2
        radius = torch.norm(P[0] - center)
        return center, radius
    else:
        A, B, C = P[0], P[1], P[2]
        a = torch.norm(B - C)
        b = torch.norm(A - C)
        c = torch.norm(A - B)
        s = (a + b + c) / 2
        area = torch.sqrt(s * (s - a) * (s - b) * (s - c))
        radius = a * b * c / (4 * area)
        center = circumcenter(A, B, C)
        return center, radius

def is_in_circle(D, p):
    center, radius = D
    return torch.norm(p - center) <= radius

def circumcenter(A, B, C):
    D = 2 * (A[0] * (B[1] - C[1]) + B[0] * (C[1] - A[1]) + C[0] * (A[1] - B[1]))
    Ux = ((A[0]**2 + A[1]**2) * (B[1] - C[1]) + (B[0]**2 + B[1]**2) * (C[1] - A[1]) + (C[0]**2 + C[1]**2) * (A[1] - B[1])) / D
    Uy = ((A[0]**2 + A[1]**2) * (C[0] - B[0]) + (B[0]**2 + B[1]**2) * (A[0] - C[0]) + (C[0]**2 + C[1]**2) * (B[0] - A[0])) / D
    return torch.tensor([Ux, Uy], device=A.device)

def minimum_enclosing_circle(points: torch.Tensor):
    device = points.device
    points = points[torch.randperm(points.size(0))].to(device)  # Shuffle the points and ensure they are on the same device
    center, radius = welzl(points, torch.empty((0, 2), dtype=points.dtype, device=device), device)
    return center, radius

