"""
Dataset column names -- the single place to adapt the code to your DataFrame.

Every embedding column the training and inference entrypoints read is named here.
If your splits use different column names, change them in this file only; nothing
else in the release hardcodes a dataset column.

The batch keys further down are internal wiring, not dataset columns: they are the
keys the datasets emit and `custom_collate` forwards to the projectors. They do not
depend on the dataset and normally need no editing.
"""

# ------------------------------ DATASET COLUMNS ------------------------------------
# Embedding columns, one per modality.
DNA_EMBEDDING_COLUMN = "cdna_seq_embedding"   # Evo2 cDNA embedding, used by both models
PST_EMBEDDING_COLUMN = "pst_embedding"        # protein structure embedding (PST model)
ESM_EMBEDDING_COLUMN = "aa_seq_embedding"     # ESM-2 amino acid embedding (ESM model)

# ------------------------------ INTERNAL BATCH KEYS --------------------------------
DNA_BATCH_KEY = "nucleotide_emb"
PST_BATCH_KEY = "structure_emb"
ESM_BATCH_KEY = "aa_emb"
