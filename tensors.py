# ============================================================================
# tensors.py - Tensor operations with plugin registry
# ============================================================================
import numpy as np
from scipy.linalg import svd
from sklearn.decomposition import PCA, NMF
from typing import Callable, Dict
import networkx as nx

from schemas import Entity
class TensorCompressor:
    """Plugin registry for tensor compression algorithms"""
    _compressors = {}
    
    @classmethod
    def register(cls, name: str):
        def decorator(func: Callable):
            cls._compressors[name] = func
            return func
        return decorator
    
    @classmethod
    def compress(cls, tensor: np.ndarray, method: str, **kwargs) -> np.ndarray:
        if method not in cls._compressors:
            raise ValueError(f"Unknown compression method: {method}")
        return cls._compressors[method](tensor, **kwargs)
    
    @classmethod
    def run_all(cls, tensor: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        return {name: func(tensor, **kwargs) for name, func in cls._compressors.items()}

@TensorCompressor.register("pca")
def pca_compress(tensor: np.ndarray, n_components: int = 8) -> np.ndarray:
    if len(tensor.shape) == 1:
        tensor = tensor.reshape(1, -1)
    pca = PCA(n_components=min(n_components, tensor.shape[1]))
    return pca.fit_transform(tensor).flatten()

@TensorCompressor.register("svd")
def svd_compress(tensor: np.ndarray, n_components: int = 8) -> np.ndarray:
    if len(tensor.shape) == 1:
        tensor = tensor.reshape(1, -1)
    U, S, Vt = svd(tensor, full_matrices=False)
    k = min(n_components, len(S))
    return (U[:, :k] @ np.diag(S[:k])).flatten()

@TensorCompressor.register("nmf")
def nmf_compress(tensor: np.ndarray, n_components: int = 8) -> np.ndarray:
    if len(tensor.shape) == 1:
        tensor = tensor.reshape(1, -1)
    tensor = np.abs(tensor)  # NMF requires non-negative
    nmf = NMF(n_components=min(n_components, tensor.shape[1]), init='random', random_state=42)
    return nmf.fit_transform(tensor).flatten()

def compute_ttm_metrics(entity: Entity, graph: nx.Graph) -> Dict[str, float]:
    """Compute Timepoint Tensor Model metrics"""
    if entity.entity_id not in graph:
        return {}
    
    metrics = {
        "eigenvector_centrality": nx.eigenvector_centrality(graph).get(entity.entity_id, 0.0),
        "betweenness": nx.betweenness_centrality(graph).get(entity.entity_id, 0.0),
        "pagerank": nx.pagerank(graph).get(entity.entity_id, 0.0),
    }
    return metrics
