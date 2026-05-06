import argparse

import numpy as np
import pandas as pd
import xarray as xr

# Wastewater biomarker variants
WW_BIOMARKERS = ["edar_biomarker_N1", "edar_biomarker_N2", "edar_biomarker_IP4"]

def get_obs_mask(dataset, var_name):
    """
    Returns a boolean mask where True = observed, False = missing/imputed.
    """
    if var_name == "wastewater":
        masks = []
        for b in WW_BIOMARKERS:
            mask_name = f"{b}_mask"
            if mask_name in dataset:
                masks.append(dataset[mask_name].values)
        if not masks:
            return None
        # A point is observed if ANY biomarker is observed
        return np.any(np.stack(masks), axis=0)
    
    mask_name = f"{var_name}_mask"
    if mask_name not in dataset:
        return None
    return dataset[mask_name].values

def compute_sparsity(obs_mask):
    """
    Sparsity = 1 - (observed / total)
    """
    if obs_mask is None:
        return np.nan
    return (1.0 - np.mean(obs_mask)) * 100

def analyze_dataset(path):
    ds = xr.open_zarr(path)
    variables = ["cases", "hospitalizations", "deaths", "wastewater"]
    
    results = {}
    for var in variables:
        mask = get_obs_mask(ds, var)
        if mask is not None:
            # If multiple runs, compute per-run then average or just overall mean?
            # User wants to compare "sparsity levels", maybe per-run distribution for synth?
            if "run_id" in ds[f"{var}_mask" if var != "wastewater" else f"{WW_BIOMARKERS[0]}_mask"].dims:
                # We can't easily index by name if it's a numpy array from .values
                # Better to use xarray ops
                if var == "wastewater":
                    masks = []
                    for b in WW_BIOMARKERS:
                        if f"{b}_mask" in ds:
                            masks.append(ds[f"{b}_mask"])
                    combined = xr.concat(masks, dim="biomarker").any(dim="biomarker")
                    run_sparsity = (1.0 - combined.mean(dim=["date", "region_id"])) * 100
                else:
                    run_sparsity = (1.0 - ds[f"{var}_mask"].mean(dim=["date", "region_id"])) * 100
                
                results[var] = {
                    "mean": float(run_sparsity.mean()),
                    "min": float(run_sparsity.min()),
                    "max": float(run_sparsity.max()),
                    "std": float(run_sparsity.std())
                }
            else:
                sparsity = compute_sparsity(mask)
                results[var] = {"mean": sparsity, "min": sparsity, "max": sparsity, "std": 0.0}
    
    ds.close()
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synth", type=str, default="data/processed/synthetic_full.zarr")
    parser.add_argument("--real", type=str, default="data/processed/real_with_id.zarr")
    args = parser.parse_args()
    
    print(f"Analyzing Synthetic: {args.synth}")
    synth_results = analyze_dataset(args.synth)
    
    print(f"Analyzing Real: {args.real}")
    real_results = analyze_dataset(args.real)
    
    comparison = []
    for var in ["cases", "hospitalizations", "deaths", "wastewater"]:
        s = synth_results.get(var, {"mean": np.nan, "min": np.nan, "max": np.nan})
        r = real_results.get(var, {"mean": np.nan})
        
        comparison.append({
            "Variable": var,
            "Synth Mean %": f"{s['mean']:.2f}%",
            "Synth Range %": f"[{s['min']:.2f}, {s['max']:.2f}]",
            "Real Mean %": f"{r['mean']:.2f}%" if not np.isnan(r['mean']) else "N/A"
        })
    
    df = pd.DataFrame(comparison)
    print("\nSparsity Comparison Table:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
