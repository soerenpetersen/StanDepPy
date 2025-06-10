from typing import List, Dict
import cobra
from dataclasses import dataclass
import re
import numpy as np
import pandas as pd

from .modeldata import ModelData

@dataclass
class SpecialistEnzymes:
    """
    Container for specialist enzymes information.

    Attributes:
        enzymes (List[str]): Unique enzyme (gene set) identifiers.
        rxns (List[str]): Reaction IDs catalyzed by each specialist enzyme.
        subSystems (List[str]): Corresponding subsystem for each reaction.
    """
    enzymes: List[str]
    rxns: List[str]
    subSystems: List[str]

@dataclass
class PromiscuousEnzymes:
    """
    Container for promiscuous enzymes information.

    Attributes:
        enzymes (List[str]): Unique enzyme (gene set) identifiers with multiple reactions.
        rxns (List[List[str]]): Lists of reaction IDs per enzyme.
        subSystems (List[List[str]]): Corresponding subsystem lists per enzyme.
        nrxns (List[int]): Number of reactions per enzyme.
    """
    enzymes: List[str]
    rxns: List[List[str]]
    subSystems: List[List[str]]
    nrxns: List[int]

@dataclass
class EnzymeData:
    """
    Container for enzyme expression comparison data.

    Attributes:
        enzyme (List[str]): Combined list of specialist and promiscuous enzyme identifiers.
        value (np.ndarray): Expression abundance matrix (enzymes x tissues).
        rxns (List[List[str]]): Corresponding reactions per enzyme.
        Tissue (List[str]): Tissue/condition names.
    """
    enzyme: List[str]
    value: np.ndarray
    rxns: List[List[str]]
    Tissue: List[str]


def parse_gpr(rule: str) -> List[List[str]]:
    """
    Parses a GPR rule string into a list of gene lists (enzyme complexes).
    E.g. "(g1 and g2) or g3" -> [["g1","g2"],["g3"]]
    """
    if not rule:
        return []
    cleaned = rule.replace('(', '').replace(')', '')
    or_split = re.split(r'\s+or\s+', cleaned, flags=re.IGNORECASE)
    complexes: List[List[str]] = []
    for part in or_split:
        genes = re.split(r'\s+and\s+', part, flags=re.IGNORECASE)
        genes = [g.strip() for g in genes if g.strip()]
        if genes:
            complexes.append(genes)
    return complexes


def get_specialist_enzymes(model: cobra.Model) -> SpecialistEnzymes:
    """
    Identify "Specialist Enzymes": gene sets that catalyze exactly one reaction.
    """
    flat_parsed, corr_rxns, corr_subsys = [], [], []
    for rxn in model.reactions:
        for comp in parse_gpr(rxn.gene_reaction_rule):
            flat_parsed.append(comp)
            corr_rxns.append(rxn.id)
            corr_subsys.append(rxn.subsystem or '')
    filtered = [(p, r, s) for p, r, s in zip(flat_parsed, corr_rxns, corr_subsys) if p]
    if not filtered:
        return SpecialistEnzymes([], [], [])
    parsed, rxns, subsys = zip(*filtered)
    parsed_str = [' & '.join(p) if len(p) > 1 else p[0] for p in parsed]
    unique_parsed = []
    for p in parsed_str:
        if p not in unique_parsed:
            unique_parsed.append(p)
    enzyme_to_rxns = {p: [] for p in unique_parsed}
    enzyme_to_subsys = {p: [] for p in unique_parsed}
    for p, r, s in zip(parsed_str, rxns, subsys):
        enzyme_to_rxns[p].append(r)
        enzyme_to_subsys[p].append(s)
    specialists, spec_rxns, spec_subsys = [], [], []
    for enzyme in unique_parsed:
        if len(enzyme_to_rxns[enzyme]) == 1:
            specialists.append(enzyme)
            spec_rxns.append(enzyme_to_rxns[enzyme][0])
            spec_subsys.append(enzyme_to_subsys[enzyme][0])
    return SpecialistEnzymes(
        enzymes=specialists,
        rxns=spec_rxns,
        subSystems=spec_subsys
    )


def get_promiscuous_enzymes(model: cobra.Model) -> PromiscuousEnzymes:
    """
    Identify "Promiscuous Enzymes": gene sets that catalyze multiple reactions.
    """
    flat_parsed, corr_rxns, corr_subsys = [], [], []
    for rxn in model.reactions:
        for comp in parse_gpr(rxn.gene_reaction_rule):
            flat_parsed.append(comp)
            corr_rxns.append(rxn.id)
            corr_subsys.append(rxn.subsystem or '')
    filtered = [(p, r, s) for p, r, s in zip(flat_parsed, corr_rxns, corr_subsys) if p]
    if not filtered:
        return PromiscuousEnzymes([], [], [], [])
    parsed, rxns, subsys = zip(*filtered)
    parsed_str = [' & '.join(p) if len(p) > 1 else p[0] for p in parsed]
    unique_parsed = []
    for p in parsed_str:
        if p not in unique_parsed:
            unique_parsed.append(p)
    enzyme_to_rxns = {p: [] for p in unique_parsed}
    enzyme_to_subsys = {p: [] for p in unique_parsed}
    for p, r, s in zip(parsed_str, rxns, subsys):
        enzyme_to_rxns[p].append(r)
        enzyme_to_subsys[p].append(s)
    prom_enzymes, prom_rxns, prom_subsys, prom_nrxns = [], [], [], []
    for enzyme in unique_parsed:
        count = len(enzyme_to_rxns[enzyme])
        if count > 1:
            prom_enzymes.append(enzyme)
            prom_rxns.append(enzyme_to_rxns[enzyme])
            prom_subsys.append(enzyme_to_subsys[enzyme])
            prom_nrxns.append(count)
    return PromiscuousEnzymes(
        enzymes=prom_enzymes,
        rxns=prom_rxns,
        subSystems=prom_subsys,
        nrxns=prom_nrxns
    )


def compare_promiscuous_specific(
    spec: SpecialistEnzymes,
    prom: PromiscuousEnzymes,
    model_data: ModelData
) -> EnzymeData:
    """
    Compare distributions of specialist vs. promiscuous enzyme expression.

    For each enzyme (specialist and promiscuous), compute enzyme abundance per tissue
    as the minimum expression across its subunit genes if all subunits are present.

    Returns:
        EnzymeData with combined enzyme identifiers, value matrix, rxns, and Tissue.
    """
    # Split enzyme identifiers into subunit lists
    spec_subs = [e.split(' & ') for e in spec.enzymes]
    prom_subs = [e.split(' & ') for e in prom.enzymes]
    tissues = model_data.Tissue
    n_tissues = len(tissues)

    # Compute specialist expression matrix
    spec_matrix = np.zeros((len(spec_subs), n_tissues), dtype=float)
    for j, subs in enumerate(spec_subs):
        # indices of subunits in model_data.gene
        idx = [model_data.gene.index(g) for g in subs if g in model_data.gene]
        if len(idx) == len(subs):
            spec_matrix[j, :] = np.min(model_data.value[idx, :], axis=0)

    # Compute promiscuous expression matrix
    prom_matrix = np.zeros((len(prom_subs), n_tissues), dtype=float)
    for j, subs in enumerate(prom_subs):
        idx = [model_data.gene.index(g) for g in subs if g in model_data.gene]
        if len(idx) == len(subs):
            prom_matrix[j, :] = np.min(model_data.value[idx, :], axis=0)

    # Combine
    combined_matrix = np.vstack([spec_matrix, prom_matrix])
    combined_rxns = ([ [r] for r in spec.rxns ] if spec.rxns else []) + prom.rxns
    combined_enzymes = spec.enzymes + prom.enzymes

    return EnzymeData(
        enzyme=combined_enzymes,
        value=combined_matrix,
        rxns=combined_rxns,
        Tissue=tissues
    )

def get_gene_to_enzym_list(enzymeData):
    """
    Return a DataFrame mapping each enzyme to its reactions.
    
    Parameters:
        enzymeData (EnzymeData): Object containing attributes
            - enzyme: List[str] of enzyme identifiers
            - rxns:   List[List[str]] of reaction lists per enzyme
    
    Returns:
        pandas.DataFrame with columns:
            - 'enzyme': enzyme identifier (string)
            - 'rxns':   semicolon-separated string of reactions
    """
    import pandas as pd

    # Convert each list of reaction IDs into a single string ("rxn1;rxn2;...")
    rxn_strings = [
        ';'.join(rxn_list) if isinstance(rxn_list, (list, tuple)) else str(rxn_list)
        for rxn_list in enzymeData.rxns
    ]

    # Build a DataFrame with one row per enzyme
    df = pd.DataFrame({
        'gene': enzymeData.enzyme,
        'rxns':   rxn_strings
    })

    return df

def translate_gene_matrix(
    expr_df: pd.DataFrame,
    translation_df: pd.DataFrame,
    mode: str = "GM1"        # GM1 = max, GM2 = sum
) -> pd.DataFrame:
    """
    Map gene-level expression to reaction-level expression
    using standard GPR aggregation rules (AND = min, OR = max or sum).

    Parameters
    ----------
    expr_df        : pd.DataFrame with rows = gene names, columns = samples
    _model         : wird nicht verwendet, aber zur Kompatibilität akzeptiert
    translation_df : pd.DataFrame mit Spalten ['enzyme', 'rxns']
                     - 'enzyme': Gen-Set, mit ' & ' getrennt (z.B. "g1 & g2")
                     - 'rxns'  : Reaktions-IDs mit ';' getrennt (z.B. "R1;R2")
    mode           : 'GM1' → OR = max  |  'GM2' → OR = sum

    Returns
    -------
    pd.DataFrame   mit Spalten ['reaction', <Sample1>, <Sample2>, …]
                   eine Zeile pro Reaktion, Reaktionsname in 'reaction'
    """
    import pandas as pd

    # --- 1. enzyme → gene list (AND) --------------------------
    enz_to_genes: Dict[str, List[str]] = {
        enz: enz.split(" & ")
        for enz in translation_df["gene"]
    }

    # --- 2. enzyme expression = min(gene) ---------------------
    enz_expr: Dict[str, pd.Series] = {}
    for enz, genes in enz_to_genes.items():
        if all(g in expr_df.index for g in genes):
            enz_expr[enz] = expr_df.loc[genes].min(axis=0)
    if not enz_expr:
        return pd.DataFrame()

    enz_expr_df = pd.DataFrame(enz_expr).T  # index = enzyme, columns = samples

    # --- 3. reaction → enzymes (OR) ---------------------------
    rxn_to_enz: Dict[str, List[str]] = {}
    for enz, rxn_str in zip(translation_df["gene"], translation_df["rxns"]):
        for rxn in rxn_str.split(";"):
            rxn_to_enz.setdefault(rxn, []).append(enz)

    # --- 4. OR-Aggregation (GM1=max, GM2=sum) ------------------
    rxn_expr: Dict[str, pd.Series] = {}
    for rxn, enz_list in rxn_to_enz.items():
        valid = [e for e in enz_list if e in enz_expr_df.index]
        if not valid:
            continue
        mat = enz_expr_df.loc[valid]
        if mode == "GM1":
            rxn_expr[rxn] = mat.max(axis=0)
        else:  # mode == "GM2"
            rxn_expr[rxn] = mat.sum(axis=0)

    # DataFrame aufbauen: index = reaction, columns = samples
    df = pd.DataFrame(rxn_expr).T
    # 'reaction'-Spalte vorne einfügen
    df.insert(0, "", df.index)
    return df


