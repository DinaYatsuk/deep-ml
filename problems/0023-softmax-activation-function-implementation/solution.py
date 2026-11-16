import math

def softmax(scores: list[float]) -> list[float]:
    # Your code here
    max_score = max(scores)
    exps = [math.exp(s - max_score) for s in scores]
    exp_sum = sum(exps)
    
    return [e / exp_sum for e in exps]