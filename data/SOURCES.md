# Data sources, provenance and terms

This repository ships two zipped third-party datasets (~15 MB) that it did not
create. They were previously committed with no attribution, no named upstream,
and no statement of licence, which is both an attribution problem and a
reproducibility problem: nothing recorded *which* version of these files the
published figures were computed from.

| File | Contents | Rows | Upstream |
|---|---|---|---|
| `data/transfers.zip` -> `top250-00-19.csv` | Top-250 most expensive football transfers per season, 2000-01 to 2018-19. Columns: `Name`, `Position`, `Age`, `Team_from`, `League_from`, `Team_to`, `League_to`, `Season`, `Market_value`, `Transfer_fee`. Values are scraped from Transfermarkt. | 4,700 | Kaggle dataset **"Football transfers 2000-2019"** by *vardan95ghazaryan* — <https://www.kaggle.com/datasets/vardan95ghazaryan/top-250-football-transfers-from-2000-to-2018> |
| `data/ratings.zip` -> `players_15.csv` … `players_20.csv`, `teams_and_leagues.csv` | FIFA video-game player ratings, editions 15-20: overall, potential, age, club, nationality, positions, contract expiry and ~70 sub-ratings per player-edition. Scraped from sofifa.com. | ~17k players per edition | Kaggle dataset **"FIFA 20 complete player dataset"** by *stefanoleone992* — <https://www.kaggle.com/datasets/stefanoleone992/fifa-20-complete-player-dataset> |

Both archives carry a file timestamp of 2019-09-26 / 2019-10-03, which is
consistent with a download made shortly after the FIFA 20 edition shipped.

## Terms

Neither dataset is first-party data. Both are **scraped** compilations of
material published by Transfermarkt and sofifa.com respectively, redistributed
on Kaggle. Practical consequences, stated plainly:

- The Kaggle pages list these datasets under open data terms, but the uploader
  is not the rights holder of the underlying Transfermarkt / sofifa content.
  Treat the files as **research/teaching use only**. Do not build a commercial
  product on them and do not redistribute them as your own.
- "FIFA", the player ratings, club names and crests are trademarks of EA Sports
  and the respective clubs. "Transfermarkt" is a trademark of Transfermarkt
  GmbH & Co. KG. No affiliation or endorsement is claimed or implied.
- The LICENSE at the repository root covers **this repository's code and
  analysis only**. It does not and cannot grant rights in the bundled data.

## Known data limitations that bear on the audit's conclusions

These are properties of the sources, not bugs in the pipeline, and they cap how
strong any finding here can be:

1. **The transfer set is truncated by fee.** It contains the *top 250 fees per
   season*, not all transfers. Every model here is therefore fitted on the
   right tail of the fee distribution, and "the model undervalues group X" means
   "within the most expensive 250 moves of each season".
2. **A fee of 0 / "NA" is not a free transfer marker you can trust.** Loans,
   undisclosed fees and free transfers are handled inconsistently upstream.
3. **FIFA ratings are opinions, not measurements.** They are produced by EA's
   scouting network and are known to lag for players outside the major European
   leagues — which is a plausible *source* of the regional disparity the audit
   measures, and is not distinguishable from model bias with this data alone.
4. **FIFA editions post-date the summer window.** Edition *Y* ships in
   September of *Y-1*. The pipeline pins each transfer season to the last
   edition released strictly before its window (see
   `fta.config.FIFA_EDITION_OFFSET`); the earlier code used the edition
   released *after* the move.
5. **Name joins are lossy and the loss is not random.** See the match-rate
   tables written to `results/` by `scripts/run_audit.py`.

## Reproducing the data

Kaggle requires an account and accepting each dataset's terms, so the archives
cannot be fetched anonymously. `scripts/fetch_data.py` verifies whatever is
present against the recorded checksums and prints the exact commands to
re-download if a file is missing or altered:

```bash
python scripts/fetch_data.py           # verify the committed archives
python scripts/fetch_data.py --print-hashes   # regenerate the manifest
```
