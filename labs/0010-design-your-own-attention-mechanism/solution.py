
import math
import numpy as np

def attention(Q, K, V):
    """
    Compute attention over a sequence.
    
    Args:
        Q: Query matrix, shape (batch_size, query_len, dim)
        K: Key matrix, shape (batch_size, key_len, dim)
        V: Value matrix, shape (batch_size, key_len, dim)
    
    Returns:
        output: Attended values, shape (batch_size, query_len, dim)
    
    The attention mechanism should:
    1. Compute compatibility between queries and keys
    2. Convert to attention weights (non-negative, sum to 1)
    3. Use weights to compute weighted sum of values
    """

    batch_size, query_len, dim = Q.shape
    _, key_len, _ = K.shape

    # Your implementation here
    # Hint: Think about how queries "ask questions" and keys "provide answers"
    
    def softmax(x, axis=-1):
        # Subtract the max along the specified axis for numerical stability
        x_shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    scores = np.matmul(Q, np.swapaxes(K, -2, -1)) / math.sqrt(dim)
    weight = softmax(scores, axis=-1)
    result = np.matmul(weight, V)
   
    return result
    
