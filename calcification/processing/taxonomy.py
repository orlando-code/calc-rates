import logging

import pandas as pd
import requests
from tqdm.auto import tqdm

from calcification.utils import config, file_ops

logger = logging.getLogger(__name__)


def assign_taxonomical_info(df: pd.DataFrame) -> pd.DataFrame:
    """Assign taxonomical information to the DataFrame."""
    df = df.assign(
        genus=df.species_types.apply(lambda x: binomial_to_genus_species(x)[0]),
        species=df.species_types.apply(lambda x: binomial_to_genus_species(x)[1]),
    )
    mapping_path = config.resources_dir / "species_mapping.yaml"
    if not mapping_path.exists():
        create_species_mapping_yaml(df.species_types.unique())
    else:
        logger.info(f"Using species mapping in {mapping_path}")
    species_mapping = file_ops.read_yaml(mapping_path)
    species_fields = ["family", "functional_group", "core_grouping"]
    for field in species_fields:
        df[field] = df.species_types.apply(
            lambda x: species_mapping.get(x, {}).get(field, "Unknown")
        )
    return df


def get_species_info_from_worms(species_binomial: str) -> dict:
    """Query WoRMS API to get the family name of a coral species.

    Args:
        species_binomial (str): Scientific name in 'Genus species' format

    Returns:
        dict: Dictionary with family name and additional taxonomic information
    """
    # strip leading/trailing whitespace
    species_binomial = species_binomial.strip()
    if any(x in species_binomial.replace(".", "").split() for x in ["sp", "spp", "cf"]):
        species_binomial = species_binomial.split()[0]
    if species_binomial in ["Massive porites", "Porites lutea/lobata"]:
        species_binomial = "Porites"
    if "CCA" in species_binomial:  # e.g. 'Unknown CCA'
        return {
            "species": species_binomial,
            "family": "Corallinaceae-Sporolithaceae",
            "status": "Best guess",
            "functional_group": "Crustose coralline algae",
            "core_grouping": "CCA",
        }

    # encode the species name to handle spaces properly in URL
    encoded_species = requests.utils.quote(species_binomial)
    base_url = f"https://www.marinespecies.org/rest/AphiaRecordsByName/{encoded_species}?like=false&marine_only=true"
    try:
        response = requests.get(base_url)
        response.raise_for_status()  # raise exception for 4XX/5XX status codes
        data = response.json()
        if data and isinstance(data, list) and len(data) > 0:
            if (
                len(data) > 1
            ):  # if more than one record, take the most recent one which has 'accepted' status
                data = [
                    record for record in data if record.get("status", "") == "accepted"
                ]
                data = data[0] if data else data[0]
            else:
                raise ValueError(f"No 'accepted' records found for {species_binomial}")
            result = {
                "species": species_binomial,
                "genus": data.get("genus", "Not Found"),
                "family": data.get("family", "Not Found"),
                "status": "Found",
                "rank": data.get("rank", "Not Found"),
                "aphia_id": data.get("AphiaID", "Not Found"),
                "accepted_name": data.get("valid_name", species_binomial),
                "kingdom": data.get("kingdom", "Not Found"),
                "phylum": data.get("phylum", "Not Found"),
                "class": data.get("class", "Not Found"),
                "order": data.get("order", "Not Found"),
            }
            result["functional_group"] = assign_functional_group(result)
            result["core_grouping"] = assign_core_groupings(result)
            return result
        else:
            return {
                "species": species_binomial,
                "family": "Not Found",
                "status": "No Data",
                "functional_group": "Unknown",
                "core_grouping": "Unknown",
            }
    except requests.exceptions.RequestException as e:
        logger.error(f"API Request Error for {species_binomial}: {e}")
        return {
            "species": species_binomial,
            "family": "Error",
            "status": str(e),
            "functional_group": "Unknown",
            "core_grouping": "Unknown",
        }


def create_species_mapping_yaml(species_list: list[str]) -> None:
    """Create a YAML file with species-family mapping for a list of species."""
    species_mapping = {}
    for species in tqdm(
        species_list, desc="Querying WoRMS API to retrieve organism taxonomic data"
    ):
        species_info = get_species_info_from_worms(species)
        species_mapping[species] = {
            "family": species_info["family"],
            "functional_group": species_info["functional_group"],
            "core_grouping": species_info["core_grouping"],
        }
    mapping_path = config.resources_dir / "species_mapping.yaml"
    file_ops.write_yaml(species_mapping, mapping_path)
    logger.info(f"Species mapping saved to {mapping_path}")


def assign_functional_group(taxon_info: dict) -> str:
    """Assign a functional group based on taxonomic information. N.B. there is likely some subjectivity in assignment"""
    family = taxon_info.get("family", "").lower()
    order = taxon_info.get("order", "").lower()
    class_name = taxon_info.get("class", "").lower()
    phylum = taxon_info.get("phylum", "").lower()
    genus = taxon_info.get("genus", "").lower()
    if genus in ["jania", "amphiroa"]:
        return "Articulated coralline algae"
    if family in [
        "corallinaceae",
        "sporolithaceae",
        "hapalidiaceae",
        "hydrolithaceae",
        "lithophyllaceae",
        "mesophyllumaceae",
        "spongitidaceae",
        "porolithaceae",
    ]:  # CCA
        return "Crustose coralline algae"
    if (
        phylum in ["chlorophyta", "ochrophyta", "rhodophyta"]
        or order in ["dictyotales", "ectocarpales", "fucales", "gigartinales"]
        or class_name in ["phaeophyceae", "ulvophyceae", "florideophyceae"]
    ):  # fleshy algae
        if genus in [
            "galaxaura",
            "padina",
            "peyssonnelia",
        ]:  # calcareous algae that aren't CCA
            return "Other algae"
        if family in ["halimedaceae"]:
            return "Halimeda"
        return "Fleshy algae"
    if order == "scleractinia" or family in [
        "pocilloporidae",
        "acroporidae",
        "poritidae",
        "faviidae",
        "fungiidae",
        "agariciidae",
    ]:  # hard coral
        return "Hard coral"
    if order in ["alcyonacea", "gorgonacea"] or "alcyoniidae" in family:  # soft coral
        return "Soft coral"
    if phylum == "porifera":  # sponge
        return "Sponge"
    if (
        phylum in ["foraminifera", "retaria"] or class_name == "foraminifera"
    ):  # foraminifera
        return "Foraminifera"
    if "turf" in taxon_info.get("species", ""):  # turf algae
        return "Turf algae"
    if phylum == "bryozoa":  # bryozoan
        return "Bryozoan"
    return "Other benthic organism"  # catchall


def assign_core_groupings(taxon_info: dict) -> str:
    """Assign a core grouping (CCA, halimeda, coral, foraminifera, other) based on taxonomical information."""
    if taxon_info.get("genus", "").lower() == "halimeda":
        return "Halimeda"
    if (
        taxon_info["functional_group"]
        in ["Crustose coralline algae", "Calcareous algae"]
        or "calcareous" in taxon_info["functional_group"].lower()
    ):
        return "CCA"
    elif taxon_info["functional_group"] in [
        "Fleshy algae",
        "Turf algae",
        "Articulated coralline algae",
    ]:
        return "Other algae"
    elif taxon_info["functional_group"] in ["Hard coral", "Soft coral"]:
        return "Coral"
    elif taxon_info["functional_group"] in ["Foraminifera"]:
        return "Foraminifera"
    else:
        return "Other"


def binomial_to_genus_species(binomial: str) -> tuple[str, str]:
    """Convert a binomial name to genus and species."""
    binomial = binomial.replace(".", "").replace("cf", "")
    split = [s for s in binomial.split(" ") if s]
    if "spp" in binomial or "sp" in split:
        genus = split[0]
        species = "spp"
    else:
        genus = split[0]
        species = split[-1] if len(split) > 1 else "spp"
    return genus, species
