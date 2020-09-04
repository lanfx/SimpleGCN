通过tensorflow实现简单的两层GCN网络用于对karate俱乐部进行分类。

参考：

[How to do Deep Learning on Graphs with Graph Convolutional Networks-1](https://towardsdatascience.com/how-to-do-deep-learning-on-graphs-with-graph-convolutional-networks-7d2250723780)

[How to do Deep Learning on Graphs with Graph Convolutional Networks-2](https://towardsdatascience.com/how-to-do-deep-learning-on-graphs-with-graph-convolutional-networks-62acf5b143d0)

版本：

python3.6

tensorflow>=2.0

注：

最后实现的效果不是很好，因为特征矩阵X采用的是单位矩阵，经过训练后只有50%的准确率。

而不经过训练有43%的准确率，暂时不知道原因。