#

- Use LightGBM instead of sklearn.ensemble.GradientBoostingClassifier
- Select parameters with "Grid search" and pursue better quality
- Select most important features, remove others and measure quality
- Use VotingClassifier and BaggingClassifier from sklearn and measure quality.
- Use mean target encoding `F' = (mean(yₖ) K + global_mean(y) α) / (K + α)`
