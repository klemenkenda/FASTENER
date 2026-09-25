# FASTENER (FeAture SelecTion ENabled by EntRopy)

## Feature Selection for Highly Dimensional Datasets

In this paper, FASTENER feature selection algorithm is presented.
The algorithm exploits entropy-based measures such as mutual information in the crossover phase of the genetic algorithm approach.
FASTENER converges to an (near) optimal subset of features faster than previous state-of-the-art algorithms and achieves better classification accuracy than similarity-based methods such as KBest or ReliefF or wrapper methods such as POSS.
The approach was evaluated using the Earth Observation dataset for land-cover classification from ESA's Sentinel-2 mission, the digital elevation model and the ground truth data of the Land Parcel Identification System from Slovenia.
The algorithm can be used in any statistical learning scenario.

Submitted to Entropy

Filip Koprivec, Klemen Kenda, Beno Šircelj
## Optional: swap operator in the pruning step (off by default)

Pruning (`purge_front_with_information_gain`) identifies a weak feature by
permutation importance and tests *deleting* it. It never tests *replacing* it,
and no other operator preserves subset size -- mating pulls a child towards the
middle size, pruning strictly decreases it, and mutation holds the size constant
only when two bits flip at once. `SwapStrategy` is that missing operator: it
drops one selected feature and inserts one unselected feature in its place, so
the candidate competes directly against the front entry at its own size.

It is opt-in. `EntropyOptimizer(..., swap_strategy=None)` -- the default -- runs
the published algorithm unchanged.

```python
from item import InformationGainSwapStrategy, RandomSwapStrategy

# Guided: drop the permutation-weakest feature, insert one sampled with
# probability weighted by mutual information.
optimizer = EntropyOptimizer(..., swap_strategy=InformationGainSwapStrategy())

# Ablation: drop a random selected feature, insert a uniformly random one.
optimizer = EntropyOptimizer(..., swap_strategy=RandomSwapStrategy())

# More than one replacement candidate per pruned subset (one fit each).
optimizer = EntropyOptimizer(
    ..., swap_strategy=InformationGainSwapStrategy(number_of_swaps=3))
```

`InformationGainSwapStrategy` reuses the mutual-information vector the mating
strategy already computed -- `EntropyOptimizer.prepare_loop` hands it over -- so
the guidance adds no scoring pass. It falls back to computing its own vector if
the mating strategy has none (e.g. `UnionMating`).

**Cost.** One extra model *fit* per proposed candidate per pruned subset. The
permutation scores it uses are already computed by the pruning step, so nothing
else is added.

**Compatibility.** `purge_item_with_information_gain(item)` keeps its name,
signature and single-`EvalItem` return value; `purge_item_candidates(item)` is
the new entry point returning the deletion candidate first and any swap
candidates after it.

## Optional: fit-capped stopping (off by default)

`Config.max_model_fits` stops the search once that many models have been fitted,
instead of running all `number_of_rounds` generations. `None` (the default)
keeps the original round-limited behaviour.

```python
config = Config(output_folder="run", random_seed=2020,
                number_of_rounds=100000,   # high: let the budget be the limit
                max_model_fits=6000)
```

It exists because comparing two variants of the algorithm is only meaningful at
equal compute, and a variant that evaluates more candidates per generation is
otherwise silently given more search. The cap is checked inside
`_fitness_function` — the single place a model is fitted — so it is exact rather
than per-generation, and nothing can spend budget unmetered. Cache hits are not
counted (they cost nothing), so the budget is one of *unique* subsets; neither
is permutation scoring, which predicts but never fits.

After a run, `optimizer.model_fits` is what was spent, `optimizer.stop_reason`
is `"fit_budget"` or `"rounds"`, and `optimizer.rounds_completed` is how many
generations finished. A generation interrupted by the cap is abandoned, and the
front is re-pruned so it holds no dominated entry.

Note that the cap must be larger than the initial population, which is evaluated
before the first generation; a smaller one raises `FitBudgetExhausted` out of
`mainloop` rather than running a search with nothing in it.

## Tests

Unit tests live in `tests/` and need only FASTENER's own requirements plus
pytest. From the repo root:

```
pip install pytest
python -m pytest tests
```
