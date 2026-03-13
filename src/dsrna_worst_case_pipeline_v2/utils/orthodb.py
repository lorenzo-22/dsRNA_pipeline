import json
import httpx
from loguru import logger
from typing import Optional

ORTHODB_BASE_URL = "https://data.orthodb.org/current"

def fetch_orthodb_data(endpoint: str, params: dict) -> Optional[dict]:
    """Helper to fetch data from OrthoDB REST API."""
    url = f"{ORTHODB_BASE_URL}/{endpoint}"
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Error fetching data from {url}: {e}")
        return None

def fetch_fasta(cluster_id: str, taxon: str, seq_type: str = "cds") -> Optional[str]:
    """Fetch FASTA sequences for a given cluster and taxon."""
    url = f"{ORTHODB_BASE_URL}/fasta"
    params = {"id": cluster_id, "species": taxon, "seqtype": seq_type}
    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            return response.text
    except Exception as e:
        logger.error(f"Error fetching FASTA for cluster {cluster_id}: {e}")
        return None

def find_cluster_for_gene(gene_id: str, gene_name: str, taxon: str) -> Optional[str]:
    """Find the OrthoDB cluster ID for a given gene ID or name."""
    search_data = fetch_orthodb_data("search", {"query": gene_id, "level": taxon})
    if not search_data or not search_data.get("data"):
        search_data = fetch_orthodb_data("search", {"query": gene_name, "level": taxon})
    if not search_data or not search_data.get("data"):
        search_data = fetch_orthodb_data("search", {"query": gene_name})
    if not search_data or not search_data.get("data"):
        return None
    if search_data.get("bigdata"):
        for entry in search_data["bigdata"]:
            cluster_id = entry.get("id")
            if cluster_id and f"at{taxon}" in cluster_id:
                return cluster_id
        return search_data["bigdata"][0].get("id")
    return search_data["data"][0]
