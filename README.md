# Repo Description

## Miscellaneous Tensor Flow layers
Currently the focus is on recommendations to leverage Tensor Flow's deep learning
framework.

* `negSamplers.py` provides a negative sampler for producing negative
examples when training ranking or classification models.

*  `ranking.py` implements a word-to-vec recommender and a
generic sigmoid-loss when one has logits and positive and negative labels.
Here also metric utilities

* `ShardedCrossEntropyWithSoftmax.py` implements a softmax where
the range of admissible labels for the examples can change. Useful
if you are training a recommender system where one
property of the query subsets the valid items, and that valid items
are indexed continuously (e.g. books in a genre). If used on non-sharded problems,
this Python version is substantially (I estimate 12X) slower than the Tensor Flow native
version (cross-entropy with logits). Read the following point for a C++ version.

* `cc/shardedXEntSfmax.cc` implements C++ versions (with & without gradient) of the
previous sharded loss. If used on non-sharded problems,
this C++ version is on par (I estimate ~20% slower) with the Tensor Flow native
version (cross-entropy with logits). Read the `cc/README.md` to compile it. The
gradient of the operation is defined in `gradShardedCrossEntropyWithSoftmax.py`. Examples
of usage & unit tests can be found in `tests/`: look for files matching `shardedCC*`.
