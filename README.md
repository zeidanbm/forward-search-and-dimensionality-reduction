# Introduction
The data we are dealing with is collected from online shoppers in an e-shop website
environment. Each row belongs to a unique user and describes a session of browsing through
the website. During the session, the user either ends up with purchase or without any purchase.
The data has 17 features and ​ 12,330 sessions(rows) collected over a period of 1 year​ . Some of those
features are categorical, ordinal or binary and we have chosen to drop them as they do not fit
well with PCA.
The data can be obtained from the UCI Machine Learning Repository through ​ this link​ . The
original authors of this data published a paper describing the use of neural networks, multilayer
perceptron and LSTM recurrent neural networks for real time predictions. (Sakar et al., 2019, 1)
Our goal here is not to come up with the best algorithm that solves the problem but to compare
the use of features extraction using forward search and dimensionality reduction using PCA.
