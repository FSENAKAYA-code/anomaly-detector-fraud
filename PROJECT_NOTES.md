# Domain implementation notes

The supervised baseline uses class-weighted random forests. Isolation Forest and One-Class SVM alternatives fit only normal training rows, select a threshold on validation labels, and report test precision/recall/F1. Unsupervised loss fields are clearly labeled score proxies.
