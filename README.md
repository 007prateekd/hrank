# hrank

This repo conatins the Python implementation of the paper [HRank: A Path based Ranking Framework in Heterogeneous Information Network](https://doi.org/10.48550/arXiv.1403.7315). Of the 3 methods mentioned in the paper, two have been implemented:

1. HRank-SY (for symmetric metapaths)
2. HRank-AS (for asymmetric metapaths)

An additional count-based method was also implemented to compare the rankings. The rankings are compared using the RBO (Rank-biased Overlap) method.

## Usage
1. Write your own metapaths in [metapaths.py](metapaths.py) in the required format or use the already existing metapaths.
2. Run `python3 hrank.py`.
3. The output would consist of the information related to the graph of metapaths, the rankings as well as the RBO score. The ranking results would also be stored in [ranks.csv](ranks.csv).
