### BindingDB unittest data

We took the first $N=100$ samples from BindingDB's tsv files using the following simple script:

```python
N=100

pairs_tsv_df = pd.read_csv("<PATH TO PAIRS TSV FILE>", sep='\t')
small_pairs_tsv = pairs_tsv_df.head(N)
small_pairs_tsv.to_csv("./small_pairs.tsv", sep="\t")

# same for ligands & targets tsv files
```