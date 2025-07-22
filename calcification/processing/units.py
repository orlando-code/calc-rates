# general
import re

import pandas as pd

# custom
from calcification import config, file_ops

### global constants
MOLAR_MASS_CACO3 = 100.0869  # g/mol
PREFIXES = {
    "c": 1e-2,
    "m": 1e-3,
    "μ": 1e-6,
    "u": 1e-6,
    "n": 1e-9,
}  # TODO: probably shouldn't need both 'μ' and 'u'
DURATIONS = {"s": 86400, "hr": 24, "d": 1, "wk": 1 / 7, "month": 365 / 12, "y": 1 / 365}


def parse_unit_components(unit: str) -> tuple[str, str]:
    """Parse a unit string into numerator and denominator components."""
    if " " not in unit:
        raise ValueError(
            f"Unit '{unit}' does not have proper format 'numerator denominator'"
        )

    components = unit.split(" ")
    if len(components) != 2:
        raise ValueError(f"Unit '{unit}' has more than two components")

    return components[0], components[1]


def extract_prefix(unit_part: str, unit_type: str) -> tuple[str, str]:
    """Extract the prefix from a unit part (e.g., 'mg' -> 'm', 'g')."""
    if "delta" in unit_part:  # relative changes
        return "", unit_part

    # determine unit type
    if unit_type == "mass":
        match = re.match(r"([nuμmcdk]?)(g|mol)", unit_part)
    elif unit_type == "area":
        match = re.match(r"([nuμmcdk]?)(m2|m-2)", unit_part)
    elif unit_type == "length":
        match = re.match(r"([nuμmcdk]?)(m{1}$)", unit_part)
    else:
        return "", unit_part

    if match:
        return match.group(1), match.group(2)
    return "", unit_part


def convert_numerator(num_part: str, rate_val: float) -> tuple[float, str]:
    """Convert the numerator part of the unit."""
    if "delta" in num_part:  # relative changes
        return rate_val, num_part

    has_caco3 = "CaCO3" in num_part
    if has_caco3:
        num_part_clean = num_part.replace("CaCO3", "")
    else:
        num_part_clean = num_part

    if "mol" in num_part_clean:  # molar units
        prefix, base = extract_prefix(num_part_clean, "mass")
        rate_val *= MOLAR_MASS_CACO3 * PREFIXES.get(prefix, 1.0)
        # convert to mg
        rate_val *= 1e3
        new_unit = "mg"
    elif "g" in num_part_clean:  # mass units
        prefix, base = extract_prefix(num_part_clean, "mass")
        rate_val *= PREFIXES.get(prefix, 1.0)
        rate_val *= 1e3  # convert to mg
        new_unit = "mg"
    elif "m2" in num_part_clean:  # area units
        prefix, base = extract_prefix(num_part_clean, "area")
        rate_val *= PREFIXES.get(prefix, 1.0) ** 2
        new_unit = "m2"
    elif re.search(r"m{1}$", num_part_clean):  # extension units
        prefix, base = extract_prefix(num_part_clean, "length")
        rate_val *= PREFIXES.get(prefix, 1.0)
        new_unit = "m"
    elif re.match(r"[nuμmcdk]{2}", num_part_clean):  # duplicate units e.g. mm
        prefix = num_part_clean[0]
        rate_val *= PREFIXES.get(prefix, 1.0)
        new_unit = num_part_clean[1:]
    else:
        new_unit = num_part_clean

    if has_caco3:  # TODO: extend to all?
        new_unit = f"{new_unit}CaCO3"

    return rate_val, new_unit


def convert_denominator(denom_part: str, rate_val: float) -> tuple[float, str]:
    """Convert the denominator part of the unit."""
    for duration, factor in DURATIONS.items():
        if duration in denom_part:
            rate_val *= factor
            denom_part = denom_part.replace(duration, "d")
            break

    if "m-2" in denom_part:  # area units
        prefix, _ = extract_prefix(denom_part, "area")
        if prefix != "c":  # convert to cm-2
            rate_val /= (PREFIXES.get(prefix, 1.0) / PREFIXES["c"]) ** 2
            denom_part = denom_part.replace(f"{prefix}m-2", "cm-2")

    elif "g" in denom_part and not (
        denom_part.startswith("d") and len(denom_part) <= 3
    ):  # mass units, avoiding 'day' confusion
        prefix, _ = extract_prefix(denom_part, "mass")
        rate_val /= PREFIXES.get(prefix, 1.0)
        denom_part = denom_part.replace(f"{prefix}g", "g")

    elif re.search(r"[nuμmcdk]{2}-2", denom_part):  # duplicate character units e.g. mm
        prefix = denom_part[0]
        rate_val /= (PREFIXES.get(prefix, 1.0) / PREFIXES["c"]) ** 2  # convert to cm-2
        denom_part = "cm-2" + denom_part[3:]

    return rate_val, denom_part


def rate_conversion(
    rate_val: float,
    rate_error: float = None,
    rate_unit: str = "",
) -> [tuple[float, str], tuple[float, str, float]]:
    """
    Convert rate to standardized units (gCaCO3 per day) and propagate errors if provided.

    Parameters:
    - rate_val: Rate value to convert
    - rate_unit: Original rate unit string (e.g., 'mgCaCO3 cm-2d-1')
    - rate_error: Standard deviation or standard error of the rate (optional)

    Returns:
    - Converted rate value
    - New standardized rate unit
    - Converted error value (if rate_error was provided)
    """
    if rate_unit is None or rate_unit != rate_unit:  # handle nans in rate_unit
        if rate_error is not None:
            return rate_val, rate_error, ""
        return rate_val, "", ""

    try:  # split into numerator and denominator
        num_part, denom_part = parse_unit_components(rate_unit)
    except ValueError as e:
        if rate_error is not None:
            return rate_val, rate_error, str(e)
        return rate_val, "", str(e)

    original_val = rate_val
    rate_val, new_num = convert_numerator(num_part, rate_val)
    rate_val, new_denom = convert_denominator(denom_part, rate_val)
    new_rate_unit = f"{new_num} {new_denom}"

    # calculate the scaling factor and propagate error if provided
    if rate_error is not None:
        if original_val != 0:
            scaling_factor = rate_val / original_val
            new_error = rate_error * abs(scaling_factor)
        else:
            # if original value is zero, can't determine scaling factor
            # In this case, we assume the error scales similarly
            new_error = rate_error
        return rate_val, new_error, new_rate_unit

    return rate_val, None, new_rate_unit


def ph_to_hplus(ph: float) -> float:
    """Convert pH to H+ concentration in μmol/kg seawater."""
    return 10 ** (-ph) * 1e6  # convert pH to H+ concentration in μmol/kg seawater


def irradiance_conversion(irr_val: float, irr_unit: str = "PAR") -> float:
    # convert from mol quanta m-2 day-1 to μmol quanta m-2 s-1
    s_in_day = DURATIONS["s"]
    return irr_val / (s_in_day * PREFIXES["μ"]) if irr_unit == "PAR" else irr_val


def map_units(df: pd.DataFrame) -> pd.DataFrame:
    """Map units to standardised units."""
    map_dict = file_ops.read_yaml(config.resources_dir / "mapping.yaml")["unit_map"]
    inverted_map = {val: key for key, values in map_dict.items() for val in values}
    df["st_calcification_unit"] = df["calcification_unit"].map(inverted_map)
    # if any calcification units are not in the map, raise error
    if df["st_calcification_unit"].isnull().any():
        missing_units = df.loc[
            df["st_calcification_unit"].isnull(), "calcification_unit"
        ].unique()
        print(f"Missing units in mapping: {missing_units}")  # TODO: replace with error

    return df
