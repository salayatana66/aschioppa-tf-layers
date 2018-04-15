# aschioppa-tf-layers

Miscellaneous Tensor Flow layers

* `negSamplers.py` provides a negative sampler for producing negative
examples when training ranking or classification models.

*  `ranking.py` implements a word-to-vec recommender and a
generic sigmoid-loss when one has logits and positive and negative labels.
Here also metric utilities

* `ShardedCrossEntropyWithSoftmax` implements a softmax where
the range of admissible labels for the examples can change. Useful
if you are training a recommender system where one
property of the query subsets the valid items, and that valid items
are indexed continuously (e.g. books in a genre). 
