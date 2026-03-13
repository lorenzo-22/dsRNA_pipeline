import re
import math
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple
from Bio import SeqIO, AlignIO
from loguru import logger
from pathlib import Path

# Try to import ViennaRNA
try:
    import RNA
    HAS_VIENNA = True
except ImportError:
    HAS_VIENNA = False

def get_gene_name(fasta_stem: str) -> str:
    """Consistently extract gene name from filename stem."""
    parts = fasta_stem.rsplit("_", 2)
    return parts[0] if len(parts) == 3 else fasta_stem

def parse_needle_output(needle_file: Path) -> Dict:
    """Parse similarity, identity and gaps from EMBOSS Needle output."""
    stats = {"Identity": 0.0, "Similarity": 0.0, "Gaps": 0.0}
    if not needle_file.exists(): return stats
    content = needle_file.read_text()
    id_match = re.search(r"Identity:\s+\d+/\d+\s+\((\d+\.\d+)%\)", content)
    sim_match = re.search(r"Similarity:\s+\d+/\d+\s+\((\d+\.\d+)%\)", content)
    gap_match = re.search(r"Gaps:\s+\d+/\d+\s+\((\d+\.\d+)%\)", content)
    if id_match: stats["Identity"] = float(id_match.group(1))
    if sim_match: stats["Similarity"] = float(sim_match.group(1))
    if gap_match: stats["Gaps"] = float(gap_match.group(1))
    return stats

def get_anchored_sequences(needle_file: Path) -> Tuple[str, str]:
    """Extract aligned sequences from Needle file and remove reference gaps."""
    try:
        alignment = AlignIO.read(needle_file, "emboss")
        ref_seq, que_seq = str(alignment[0].seq), str(alignment[1].seq)
        anchored_ref, anchored_que = [], []
        for r, q in zip(ref_seq, que_seq):
            if r != '-':
                anchored_ref.append(r)
                anchored_que.append(q)
        return "".join(anchored_ref), "".join(anchored_que)
    except Exception as e:
        logger.error(f"Error parsing sequences from {needle_file}: {e}")
        return "", ""

def run_vienna_accessibility(sequence: str, window_size: int = 150, max_span: int = 100) -> np.ndarray:
    """Calculate per-nucleotide accessibility (u=1) using ViennaRNA."""
    if not HAS_VIENNA:
        logger.warning("ViennaRNA not found. Returning zero accessibility.")
        return np.zeros(len(sequence))
    try:
        seq_len = len(sequence)
        rna_seq = sequence.replace("T", "U").replace("t", "u")
        w = min(window_size, seq_len)
        l = min(max_span, seq_len)
        probs_matrix = RNA.pfl_fold_up(rna_seq, 1, w, l)
        acc = np.zeros(seq_len)
        for i in range(1, seq_len + 1):
            try:
                acc[i-1] = probs_matrix[i][1]
            except (IndexError, TypeError):
                acc[i-1] = 0.0
        return acc
    except Exception as e:
        logger.error(f"Error calculating accessibility: {e}")
        return np.zeros(len(sequence))

def calculate_column_ic(column: List[str]) -> float:
    """Calculate Shannon IC for a single alignment column."""
    valid_bases = [b.upper() for b in column if b.upper() in 'ACGTU']
    n = len(valid_bases)
    if n == 0: return 0.0
    s, h_max, ln2 = 4, 2.0, math.log(2)
    counts = {b: valid_bases.count(b) for b in 'ACGT'}
    if 'U' in valid_bases: counts['T'] += valid_bases.count('U')
    h_l = 0
    for b in 'ACGT':
        p = counts[b] / n
        if p > 0: h_l -= p * math.log2(p)
    e_n = (s - 1) / (2 * n * ln2)
    return max(0, h_max - (h_l + e_n))

def calculate_information_content(alignment_file: Path, reference_id: Optional[str] = None) -> pd.DataFrame:
    """Calculate IC per alignment position, mapped to reference."""
    records = list(SeqIO.parse(alignment_file, "fasta"))
    if not records: return pd.DataFrame()
    aln_len = len(records[0].seq)
    ref_seq = next((str(r.seq) for r in records if reference_id == r.id), None)
    results = []
    ref_pos = 0
    for i in range(aln_len):
        col = [str(r.seq[i]) for r in records]
        is_ref_gap = False
        if ref_seq:
            if ref_seq[i] != '-': ref_pos += 1
            else: is_ref_gap = True
        results.append({
            "Position": i + 1, 
            "ReferencePosition": ref_pos if not is_ref_gap else None,
            "IC": calculate_column_ic(col)
        })
    return pd.DataFrame(results)

def add_labels_to_barplot(ax):
    """Add percentage labels above bars in a barplot."""
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}%', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 9), 
                        textcoords = 'offset points',
                        fontsize=8)
