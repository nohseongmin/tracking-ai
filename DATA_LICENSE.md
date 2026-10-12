# Data and result licensing

## Source dataset

- Title: Esports Trajectories and Actions (ESTA)
- Creators: Xenopoulos et al.; maintained at `pnxenopoulos/esta`
- Source: https://github.com/pnxenopoulos/esta
- Paper: https://arxiv.org/abs/2209.09861
- License: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
- License text: https://creativecommons.org/licenses/by-sa/4.0/legalcode.en

ESTA is downloaded separately. Its original license and attribution remain applicable to the dataset and any distributed adaptations. The repository's MIT code license does not relicense ESTA.

## Published results

Files under `results/` and `assets/accuracy_overview.png` are released by nohseongmin under **CC BY-SA 4.0**. They summarize experiments performed on ESTA; they do not reproduce player-level recordings.

Changes from the source material include selecting continuous alive-player movement windows, constructing future-displacement targets, fitting prediction models, calculating aggregate errors and threshold success rates, and plotting the results. Experimental configurations were created by this project.

Suggested attribution:

> Creature Tracking AI experiment reports by nohseongmin, based on the Esports Trajectories and Actions (ESTA) dataset by Xenopoulos et al. Source: https://github.com/pnxenopoulos/esta. Reports and figures: CC BY-SA 4.0, https://creativecommons.org/licenses/by-sa/4.0/.

Original Python code, configuration-generation logic, and authored documentation use the MIT license in `LICENSE`. Trained weights are not distributed with this repository. No additional licensing claim about trained weights is made here.
