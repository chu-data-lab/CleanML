"""Define the domain of scenarios"""
from .error_type import *

scenarios = {
    "missing_values":["CD"],
    "outliers":["BD", "CD"],
    "mislabel":["BD", "CD"],
    "inconsistency": ["BD", "CD"],
    "duplicates": ["BD", "CD"]
}