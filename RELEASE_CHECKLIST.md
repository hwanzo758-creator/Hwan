# Release Checklist

Before making this repository public, check the following items.

## Required

- [x] Replace `https://github.com/<owner>/<repository>` in `CITATION.cff` after the GitHub repository is created.
- [ ] Decide whether the exact manuscript `scene_index.csv` can be redistributed.
- [ ] If redistribution is allowed, add `scene_index.csv` at the repository root and confirm it matches `DATASET_DIR/index/scene_index.csv` used in the manuscript.
- [ ] If redistribution is not allowed, keep `scene_index_template.csv` and update the manuscript/Data Availability wording so it does not promise a released exact scene-index file.
- [ ] Make the manuscript code-availability wording consistent: use "has been released" and cite https://github.com/hwanzo758-creator/UAV-Multispectral-Super-Resolution if the repository is public.
- [ ] Confirm that no raw AI Hub data, preprocessed cubes, checkpoints, generated SR outputs, or large figures are committed.
- [ ] Confirm that `Final_version5_revision.py` uses placeholder paths only, not private local paths.

## Recommended

- [ ] Add the final paper DOI to `CITATION.cff` after publication.
- [ ] Add a short GitHub repository description: `UAV five-band multispectral super-resolution revision code with HAT/DAT, spectral metrics, VI preservation, degradation and noise robustness.`
- [ ] Tag the release as `v1.0.0` after the repository is public and manuscript wording is final.