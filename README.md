# FASTENER (FeAture SelecTion ENabled by EntRopy)

In this paper, FASTENER feature selection algorithm is presented.
The algorithm exploits entropy-based measures such as mutual information in the crossover phase of the genetic algorithm approach.

FASTENER converges to an (near) optimal subset of features faster than previous state-of-the-art algorithms and achieves better classification accuracy than similarity-based methods such as KBest or ReliefF or wrapper methods such as POSS.

The approach was evaluated using the Earth Observation dataset for land-cover classification from ESA's Sentinel-2 mission, the digital elevation model and the ground truth data of the Land Parcel Identification System from Slovenia.

The algorithm can be used in any statistical learning scenario.

## Changes since the published version

The algorithm has changed since the 2020 paper: new optional operators, and bug
fixes that change the default search. See [CHANGELOG.md](CHANGELOG.md) for what
changed, when, and how to reproduce the published behaviour.

## Tests

Unit tests live in `tests/` and need only FASTENER's own requirements plus
pytest. From the repo root:

```
pip install pytest
python -m pytest tests
```

## Citation

If you use FASTENER, please cite the paper:

> Koprivec, F.; Kenda, K.; Šircelj, B. FASTENER Feature Selection for Inference
> from Earth Observation Data. *Entropy* **2020**, 22(11), 1198.
> [doi:10.3390/e22111198](https://doi.org/10.3390/e22111198)

```bibtex
@article{koprivec2020fastener,
  author  = {Koprivec, Filip and Kenda, Klemen and {\v{S}}ircelj, Beno},
  title   = {{FASTENER} Feature Selection for Inference from Earth Observation Data},
  journal = {Entropy},
  volume  = {22},
  number  = {11},
  pages   = {1198},
  year    = {2020},
  doi     = {10.3390/e22111198}
}
```

The Earth observation dataset used in the paper (Sentinel-2 land patch samples
over Slovenia, 2017) is published separately:

> Šircelj, B.; Kenda, K.; Koprivec, F. Land Patch Samples. *PANGAEA* **2020**.
> [doi:10.1594/PANGAEA.914271](https://doi.org/10.1594/PANGAEA.914271) (CC-BY-4.0)

```bibtex
@misc{sircelj2020landpatch,
  author    = {{\v{S}}ircelj, Beno and Kenda, Klemen and Koprivec, Filip},
  title     = {Land Patch Samples},
  publisher = {PANGAEA},
  year      = {2020},
  doi       = {10.1594/PANGAEA.914271}
}
```
