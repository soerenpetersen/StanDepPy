from src import data
from src import modeldata
from src import enzymes
from src import clustering
from src import utils
#from src import models
import numpy as np
import show
import pandas as pd

# True: Bcells
# False: Celegans
dataset = False
# column index in metadata CSV for cell types (0-based)
cell_type_column = 1  # z.B. Spalte 1 enthält Zelltypen

if not dataset:
    # Celegans
    model_path = "data/model/iCEL1314.xml"
    c_path = "data/core_files/celegans/caenorhabditis.elegans.SCE.QCed_qc.mtx"
    g_path = "data/core_files/celegans/caenorhabditis.elegans.SCE.QCed_features.csv"
    s_path = "data/core_files/celegans/caenorhabditis.elegans.SCE.QCed_metadata.csv"

if dataset:
    # Bcells
    model_path = "data/model/Recon3DModel_301.xml"
    c_path = "data/core_files/bcells/count_matrix.mtx"
    g_path = "data/core_files/bcells/rownames_translated.csv"
    s_path = "data/core_files/bcells/colnames.csv"

# Lade ExpressionData und Metadata (Zelltypen)
expr = data.load_expression_data(
    counts_path=c_path,
    genes_path=g_path,
    samples_path=s_path,
    cell_type_column=cell_type_column
)

# SBML-Modell einlesen
model = modeldata.load_model(model_path)
# ExpressionData auf Modell-Gene matchen
model_data = modeldata.get_model_data(expr, model)

print(f"Genes present: {len(model_data.ID_genePresent)}")
print(f"Genes missing: {len(model_data.ID_geneMissing)}")

# Enzym-Klassifikation
spec = enzymes.get_specialist_enzymes(model)
prom = enzymes.get_promiscuous_enzymes(model)
print(f"Spec: {len(spec.enzymes)}")
print(f"Prom: {len(prom.enzymes)}")

# Enzym-Expression berechnen
enzymeData = enzymes.compare_promiscuous_specific(spec, prom, model_data)
print(f"Enzymes: {len(enzymeData.enzyme)}")

enzyme_gene_list = enzymes.get_gene_to_enzym_list(enzymeData)

enzyme_gene_list.to_csv('enzyme_gene_list.csv', index=False)

bootstrapped_mtx = data.load_expression_csv("caenorhabditiselegansSCEQCed_full_mtx.csv")

rxn_matrix = enzymes.translate_gene_matrix(bootstrapped_mtx, enzyme_gene_list, mode="GM1")

utils.plot_reaction_distribution(rxn_matrix, "rxn_distribution.png", bins="auto", log10=True, include_zeros=False)

rxn_matrix.to_csv('rxn_matrix.csv', index=False)

edges = [-2, -1, 0, 1, 2, 2.5, 3, 4]

res, adj_edges = clustering.bin_reaction_expression(rxn_matrix, edges, adjust_bins=True)

res_norm = clustering.norm_bin_counts(res)

res_norm.to_csv('binned_rxn_matrix.csv', index=False)

print(adj_edges)

#clustering.plot_jaccard_curve(res_norm, max_k=100, save_path="jaccard_curve.png")

df_clusters, chosen_bin = clustering.cluster_reactions_iterative(res_norm, rxn_matrix, max_k=100, similarity_threshold=0.9)

df_clusters.to_csv('clustered_reactions.csv', index=False)
chosen_bin.to_csv('cluster_stats.csv', index=False)

#print(out)





# Parameter für Clustering
# edgeX = np.array([
#     0.0,  # kein Expression
#     0.1, 0.2, 0.3, 0.4, 0.5,  # sehr niedrig
#     0.7, 1.0,                 # gering bis moderat
#     1.5, 2.0,                 # moderat bis hoch
#     2.5, 3.0,                 # hoch
#     4.0, 5.0,                 # sehr hoch (falls vorhanden)
#     6.0                       # oberes Ende
# ])
# k = 30  # Anzahl Cluster
# clustObj, Z_ord, dendro = clustering.gene_expr_dist_hierarchy(
#     enzymData,
#     remove_objects=None,
#     edgeX=edgeX,
#     k=k,
#     dist_method='euclidean',
#     linkage_method='complete',
#     plot=True
# )

# # Tissue-Namen für Modelle: nutze Zelltypen aus expr, wenn vorhanden
# tisNames = expr.cell_types if hasattr(expr, 'cell_types') and expr.cell_types is not None else expr.Tissue


# # Visualisierung
# show.summarize_and_plot_clustering(clustObj)

# geneTis, enzTis, cutOff, thr, enzSel, tisClust = models.models4m_clusters1(
#     clustObj=clustObj,
#     tisNames=tisNames,
#     model=model,
#     xedge=edgeX,
#     folderName="plots",
#     cutOff=None,
#     figFlag=False,
#     adjustValue=0.0,
#     weightage=[1.0, 1.0]
# )

# print("Cutoff percentiles per cluster:", cutOff)
# print("Log10 thresholds per cluster:", thr)
# # geneTis is now your active-reaction mask if model was given

# df_clusters = pd.DataFrame({
#     "cluster_index": np.arange(1, len(cutOff) + 1),
#     "cutOff_percentile": cutOff,
#     "log10_threshold": thr
# })
# df_clusters.to_csv("plots/cluster_stats.csv", index=False)

# # Angenommen clustObj.objects enthält die Objekt-Namen (Gene oder Enzyme)
# # und tisNames ist die Liste der Tissue-Namen
# df_geneTis = pd.DataFrame(geneTis, index=clustObj.objects, columns=tisNames)
# df_geneTis.to_csv("plots/geneTis_matrix.csv")

# df_enzTis = pd.DataFrame(enzTis, index=clustObj.objects, columns=tisNames)
# df_enzTis.to_csv("plots/enzTis_matrix.csv")

# df_enzSel = pd.DataFrame(enzSel, index=clustObj.objects, columns=tisNames)
# df_enzSel.to_csv("plots/enzSel_matrix.csv")

# # tisClust enthält u. a. "outperm" (Reihenfolge der Tissues) und "names" (Tissue-Namen in Reordered-Form).
# df_tisClust = pd.DataFrame({
#     "original_index": np.arange(len(tisNames)),
#     "reordered_index": tisClust["outperm"],
#     "reordered_name": tisClust["names"]
# })
# df_tisClust.to_csv("plots/tissue_clustering.csv", index=False)

