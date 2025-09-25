# veho_net/sort_optimization.py - SIMPLIFIED VERSION (sort optimization disabled)
import pandas as pd
import numpy as np


def calculate_containerization_costs_corrected(od_selected: pd.DataFrame, facilities: pd.DataFrame,
                                               mileage_bands: pd.DataFrame, costs: dict,
                                               timing_kv: dict, package_mix: pd.DataFrame,
                                               container_params: pd.DataFrame) -> pd.DataFrame:
    """
    SIMPLIFIED: Return empty DataFrame to disable sort optimization for now.
    This allows the core functionality to work while we fix sort optimization separately.
    """
    print("INFO: Sort optimization is disabled in this simplified version")
    return pd.DataFrame()


def optimize_sort_allocation(od_selected: pd.DataFrame, cost_analysis: pd.DataFrame,
                             facilities: pd.DataFrame, timing_kv: dict) -> pd.DataFrame:
    """SIMPLIFIED: Return empty DataFrame."""
    return pd.DataFrame()


def apply_sort_allocation(od_selected: pd.DataFrame, sort_allocation: pd.DataFrame,
                          cost_analysis: pd.DataFrame, facilities: pd.DataFrame) -> pd.DataFrame:
    """SIMPLIFIED: Return OD data unchanged."""
    return od_selected


def enhanced_container_truck_calculation(lane_od_pairs, package_mix, container_params, cost_kv, strategy):
    """
    SIMPLIFIED: Basic truck calculation for core functionality.
    """
    total_pkgs = sum(pkgs for _, pkgs in lane_od_pairs)

    if total_pkgs <= 0:
        return {
            'trucks_needed': 1,
            'truck_fill_rate': 0.8,
            'container_fill_rate': 0.8,
            'packages_dwelled': 0,
            'physical_containers': 1
        }

    # Simple calculation - assume 2000 packages per truck capacity
    trucks_needed = max(1, int(np.ceil(total_pkgs / 2000)))

    return {
        'trucks_needed': trucks_needed,
        'truck_fill_rate': min(1.0, total_pkgs / (trucks_needed * 2000)),
        'container_fill_rate': min(1.0, total_pkgs / (trucks_needed * 2000)),
        'packages_dwelled': max(0, total_pkgs - (trucks_needed * 2000)),
        'physical_containers': trucks_needed * 4 if strategy.lower() == 'container' else 0
    }