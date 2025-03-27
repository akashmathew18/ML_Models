"""Input validation utilities for the ML Models application."""

from typing import Dict, Any, Tuple, Union

def validate_input(data: Dict[str, Any], model_type: str) -> Tuple[bool, Union[Dict[str, float], str]]:
    """Validate input data for model predictions.
    
    Args:
        data: Dictionary containing input features
        model_type: Type of model ('SLR', 'MLR', 'LR', 'PR', 'kNN')
    
    Returns:
        Tuple of (is_valid, processed_data or error_message)
    """
    try:
        if model_type == "SLR":
            if "training_hours" not in data:
                return False, "Missing required field: training_hours"
            hours = float(data["training_hours"])
            if hours < 0:
                return False, "Training hours cannot be negative"
            return True, {"training_hours": hours}

        elif model_type == "MLR":
            required_fields = ["training_hours", "fitness_level", "past_performance"]
            for field in required_fields:
                if field not in data:
                    return False, f"Missing required field: {field}"
            
            processed = {}
            for field in required_fields:
                value = float(data[field])
                if value < 0:
                    return False, f"{field.replace('_', ' ').title()} cannot be negative"
                if field in ["fitness_level", "past_performance"] and value > 100:
                    return False, f"{field.replace('_', ' ').title()} cannot exceed 100"
                processed[field] = value
            return True, processed

        elif model_type in ["LR", "PR"]:
            required_fields = ["matches", "goals", "assists", "fitness"]
            for field in required_fields:
                if field not in data:
                    return False, f"Missing required field: {field}"
            
            processed = {}
            for field in required_fields:
                value = float(data[field])
                if value < 0:
                    return False, f"{field.title()} cannot be negative"
                if field == "fitness" and value > 100:
                    return False, "Fitness score cannot exceed 100"
                processed[field] = value
            return True, processed

        elif model_type == "kNN":
            required_fields = ["matches", "goals", "assists", "fitness", "attendance", "rating"]
            for field in required_fields:
                if field not in data:
                    return False, f"Missing required field: {field}"
            
            processed = {}
            for field in required_fields:
                value = float(data[field])
                if value < 0:
                    return False, f"{field.title()} cannot be negative"
                if field in ["fitness", "attendance"] and value > 100:
                    return False, f"{field.title()} score cannot exceed 100"
                if field == "rating" and value > 5:
                    return False, "Rating cannot exceed 5"
                processed[field] = value
            return True, processed

        else:
            return False, f"Unknown model type: {model_type}"

    except ValueError:
        return False, "All input fields must be valid numbers"
