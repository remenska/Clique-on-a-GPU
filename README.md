# Clique-on-a-GPU
This is not Bron-Kerbosch or one of its enhancements, but a much simpler algorithm.

Deploy a simplified clique algorithm on a GPU

For each item in the list, check its correlation with each of the other items.
Remove the item with the least number of correlations.
This will reduce the number of correlations - with 1 - for a number of other items in the list.
Again, find the item with the least number of correlations and remove it.
And so on.


