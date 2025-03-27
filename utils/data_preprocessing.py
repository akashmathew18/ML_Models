"""Data preprocessing utilities for the ML Models application."""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

def preprocess_data(
    data: Dict[str, float],
    scaler: Optional[StandardScaler] = None,
    transformer: Optional[PolynomialFeatures] = None
) -> Tuple[np.ndarray, Optional[str]]:
    """Preprocess input data for model prediction.
    
    Args:
        data: Dictionary containing input features
        scaler: Optional StandardScaler for feature scaling
        transformer: Optional PolynomialFeatures transformer
    
    Returns:
        Tuple of (processed_data, error_message if any)
    """
    try:
        # Convert dictionary to numpy array
        X = np.array(list(data.values())).reshape(1, -1)
        
        # Apply polynomial transformation if provided
        if transformer is not None:
            X = transformer.transform(X)
            
        # Apply scaling if provided
        if scaler is not None:
            X = scaler.transform(X)
            
        return X, None
        
    except Exception as e:
        return None, f"Error in data preprocessing: {str(e)}"
