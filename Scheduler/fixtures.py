"""Explicit synthetic arrivals and class labels, not image inference."""
from INFRA.contracts import Job

def demonstration_jobs():
    return [
        Job('job-0001', 'synthetic-image-001', 'ue-001', 0, 60, 2, 100),
        Job('job-0002', 'synthetic-image-002', 'ue-002', 0, 60, 0, 30),
        Job('job-0003', 'synthetic-image-003', 'ue-001', 0, 40, 0, 20),
        Job('job-0004', 'synthetic-image-004', 'ue-003', 10, 60, 1, 60),
        Job('job-0005', 'synthetic-image-005', 'ue-002', 20, 40, 0, 40),
        Job('job-0006', 'synthetic-image-006', 'ue-003', 30, 100, 2, 120),
    ]
