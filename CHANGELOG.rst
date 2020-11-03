Changelog
=========


v0.1.3 (2020-11-03)
-------------------

Changes
~~~~~~~
- Chore: preparing pypi release. [Kevin M. Jablonka]
- Chore: fixing broken links in readme after rename. [Kevin M. Jablonka]
- Chore: renaming the package. [Kevin M. Jablonka]

  BREAKING CHANGE
- Chore: preparing release to pypi. [Kevin M. Jablonka]
- Chore: remove pypi instructions due to clashing name. [Kevin M.
  Jablonka]

Other
~~~~~
- Merge branch 'master' of github.com:kjappelbaum/PyPAL. [Kevin M.
  Jablonka]


v0.1.2 (2020-11-03)
-------------------

Changes
~~~~~~~
- Chore: preparing release to pypi. [Kevin M. Jablonka]
- Chore: bug in the postBuild script was a space. [Kevin M. Jablonka]
- Chore: remove space in postBuild script. [Kevin M. Jablonka]
- Chore: for binder we need to install pypal. [Kevin M. Jablonka]
- Chore: updated configs for readthedocs and binder. [Kevin M. Jablonka]
- Chore: making readthedocs ready. [Kevin M. Jablonka]

Docs
~~~~
- Docs: move doc links to readthedocs. [Kevin M. Jablonka]


v0.1 (2020-10-25)
-----------------

New
~~~
- Feat: Implemented first version of PALGBDT (#75) [Kevin Jablonka]

  * feat: started implementation of PALGBDT

  * added framework for class

  * added helper function to construct model tuples

  * chore: implemented input validation for LGBMRegressor

  * docs: example of quantile regression

  * chore: API docs for PALGBDT, added validation for interquartile_scaler

  * chore: drop n_jobs support for palgbdt
- Feat: implemented kwargs for filtering based on the coefficient of
  variation, closes #58. [Kevin M. Jablonka]
- Feat: added new example notebook, closes #30 (#61) [Kevin Jablonka]
- Feat: warning for too low variance, closes #45 and #50. [Kevin M.
  Jablonka]
- Feat: implementing cross validation routine to address #45 (#49)
  [Kevin Jablonka]
- Feat: Adding plotting subpackage (#48) [Kevin Jablonka]

  * feat: addressing #23

  * feat: addressing #23

  * fix: normalizing histogram, closes #35

  * chore: used scaled uncertain for plot
- Feat: added uncertainty wts property (#42) [Kevin Jablonka]
- Feat: Implement mulitprocessing support for GPy and sklearn (#41)
  [Kevin Jablonka]

  * feat: first multiprocessing implementation for sklearn, #36

  * feat: first multiprocessing implementation for GPy and added note in docs, #36

  * fix: more restarts for sklearn

  * chore: fix random seed for tests
- Feat: Batch sampling (#33) [Kevin Jablonka]

  - we decided to update the mask array with the sampled points only in the `update_train_set()` function, which makes sense to me as before that the sampling didn't really happen. This is, a sample without subsequent measurement does not really help us and we do not know if only some or all objectives have been measured
  - for this reason, we implement it via `exclude_idx` in the `sample` function.
- Feat: first steps to make it nan-compatible, addressing #24. [Kevin M.
  Jablonka]
- Feat: first steps to make it nan-compatible, addressing #24. [Kevin M.
  Jablonka]
- Feat: first steps to make it nan-compatible, addressing #24. [Kevin M.
  Jablonka]
- Feat: use the actual sampled data and not the GPR predictions.
  Adressing #7. [Kevin]
- Feat: added hv indicator, closes #16. [Kevin]
- Feat: added sampling utils. [Kevin]
- Feat: renaming package, addressing #5. [Kevin]
- Feat: renaming package, addressing #5. [Kevin]
- Feat: renaming package, addressing #5. [Kevin]
- Feat: renaming package, addressing #5. [Kevin]
- Feat: renaming package, addressing #5. [Kevin]
- Feat: renaming package, addressing #5. [Kevin]
- Feat: renaming package, addressing #5. [Kevin]
- Feat: PALCoregionalized implemented. [Kevin]
- Feat: implemented turn_to_maximization. [Kevin]
- Feat: implemented update_train_set function. [Kevin]
- Feat: implemented PALSklearn. [Kevin]
- Feat: PALGPy class. [Kevin]

Changes
~~~~~~~
- Chore: fix pre-commit. [Kevin M. Jablonka]
- Chore: updating doc notes about pre-commit and adding notes to readme.
  [Kevin M. Jablonka]
- Chore: license badge. [Kevin M. Jablonka]
- Chore: added license, closes #60. [Kevin M. Jablonka]
- Chore: added license, closes #60. [Kevin M. Jablonka]
- Chore: added license, closes #60. [Kevin M. Jablonka]
- Chore: devops, docs, and closing #79. [Kevin M. Jablonka]
- Chore: added some notebooks to create the figures in the docs. [Kevin
  M. Jablonka]
- Chore: updating classifiers in setup.py, closes #73. [Kevin M.
  Jablonka]
- Chore: implemented coefficient of variation mask. [Kevin M. Jablonka]
- Chore: spelling of PyPAL in doc landing page fixed. [Kevin M.
  Jablonka]
- Chore: adding some test for epsilon sensitivity, closes #54 (#64)
  [Kevin Jablonka]
- Chore: docs example for overconfident model, fized errorbars in the
  plotting functions. [Kevin M. Jablonka]
- Chore: tuning the plotting functions. [Kevin M. Jablonka]

  * specify zorder
  * labels lowercase
- Chore: adding more tests to address #59. [Kevin M. Jablonka]
- Chore: add logger and remove print statement. [Kevin M. Jablonka]
- Chore: disablying cross-validation in  most tests, closes #53 (#55)
  [Kevin Jablonka]
- Chore: reducing restarts, fix random seed, closes #52. [Kevin M.
  Jablonka]
- Chore: fixing typo in prospector settings. [Kevin M. Jablonka]
- Chore: test with different kernel type, #37. [Kevin M. Jablonka]
- Chore: allow for None option in tests. [Kevin M. Jablonka]
- Chore: binh-korn test with smaller beta scale. [Kevin M. Jablonka]
- Chore: added more binh-korn tests to address #37. [Kevin M. Jablonka]
- Chore: added more binh-korn tests to address #37. [Kevin M. Jablonka]
- Chore: added more binh-korn tests to address #37. [Kevin M. Jablonka]
- Chore: testing binh korn. [Kevin M. Jablonka]
- Chore: added docs to PR template, closes #32. [Kevin M. Jablonka]
- Chore: updating docs #29. [Kevin M. Jablonka]
- Chore: updating docs #29. [Kevin M. Jablonka]
- Chore: updating docs #29. [Kevin M. Jablonka]
- Chore: updating docs #29. [Kevin M. Jablonka]
- Chore: updating docs #29. [Kevin M. Jablonka]
- Chore: work on lvmogp on seperate branch. [Kevin M. Jablonka]
- Chore: added missing  data tests. [Kevin M. Jablonka]
- Chore: adding tests for missing data. [Kevin M. Jablonka]
- Chore: updating missing data information in docs. [Kevin M. Jablonka]
- Chore: trigger CI. [Kevin M. Jablonka]
- Chore: added dependabot. [Kevin M. Jablonka]
- Chore: adding authors, closes #12. [Kevin M. Jablonka]
- Chore: pinning dependencies, closes #3. [Kevin M. Jablonka]
- Chore: fixing uncertainty typos. closes #22. [Kevin M. Jablonka]
- Chore: updating docs. [Kevin]
- Chore: updating docs. [Kevin]
- Chore: updating docs. [Kevin]
- Chore: updating docs. [Kevin]
- Chore: updating docs. [Kevin]
- Chore: updating docs. [Kevin]
- Chore: updating docs. [Kevin]
- Chore: updatings docs, changing API of samplign utilities - in
  practice it is probably better to not take y in the sampling utilities
  - writing more detailed docs on how to use the code. [Kevin]
- Chore: updated readme. [Kevin]
- Chore: added test for _replace_by_measurements. [Kevin]
- Chore: home logo for docs, closes #20. [Kevin]
- Chore: changed sphinx themes, closes #19 also working on testing #18.
  [Kevin]
- Chore: changed sphinx themes, closes #19 also working on testing #18.
  [Kevin]
- Chore: added scale invariance test, closes #17. [Kevin]
- Chore: updating readme. [Kevin]
- Chore: added test for minimization, closing #15. [Kevin]
- Chore: added example notebook. [Kevin]
- Chore: preparing readthedocs. [Kevin]
- Chore: updated train data. [Kevin]
- Chore: adding tests. [Kevin]
- Chore: docs. [Kevin]
- Chore: docs. [Kevin]
- Chore: docs. [Kevin]
- Chore: docs. [Kevin]
- Chore: reverting the addition of CI for docs. [Kevin]

  - Maybe it is better to build locally?
  - But we should anyhow serve them on GitHub pages
- Chore: trying to add CI for docs #9. [Kevin]
- Chore: added docs. [Kevin]
- Chore: adding test cases. [Kevin]
- Chore: adding test cases. [Kevin]
- Chore: running one step test of binh korn. [Kevin]
- Chore: running one step test of binh korn. [Kevin]
- Chore: added bihn korn test function as fixture. [Kevin]
- Chore: updated sampling #6. [Kevin]
- Chore: adding tests. [Kevin]
- Chore: adding tests. [Kevin]
- Chore: adding tests. [Kevin]
- Chore: making stronger test cases. [Kevin]
- Chore: adding tests. [Kevin]
- Chore: adding tests. [Kevin]
- Chore: update contribution guide. [Kevin]
- Chore: updated readme. [Kevin]
- Chore: testing beta update. [Kevin]
- Chore: added tests. [Kevin]
- Chore: adding tests. [Kevin]
- Chore: added tests. [Kevin]
- Chore: added tests. [Kevin]
- Chore: added tests. [Kevin]
- Chore: added tests. [Kevin]
- Chore: adding tests. [Kevin]
- Chore: disabling numba for coverage report. [Kevin]
- Chore: adding tests. [Kevin]
- Chore: adding tests. [Kevin]
- Chore: adding tests. [Kevin]
- Chore: adding tests. [Kevin]
- Chore: adding tests. [Kevin]
- Chore: adding tests. [Kevin]
- Chore: adding tests. [Kevin]
- Chore: updating coveragerc. [Kevin]
- Chore: scaled logo. [Kevin]
- Chore: adding tests. [Kevin]
- Chore: added rc file for coverage. [Kevin]
- Chore: added code coverage. [Kevin]
- Chore: adding more test cases. [Kevin]
- Chore: smaller logo. [Kevin]
- Chore: added logo placeholder. [Kevin]
- Chore: updating readme. [Kevin]
- Chore: drop Python 3.5 support due to close EOL. [Kevin]
- Chore: for now, skipping prospector in the CI: [Kevin]

  - I do not want to install the dependencies in the pre-commit workflow
  - We can run prospector after pytest in the python_package workflow
- Chore: updating README. [Kevin]
- Chore: updating README. [Kevin]
- Chore: updating pre-commit workflow. [Kevin]
- Chore: updating pre-commit workflow. [Kevin]
- Chore: updating pre-commit workflow. [Kevin]
- Chore: added CI. [Kevin]
- Chore: updating readme to use sklearn as example for subclassing.
  [Kevin]
- Chore: updated acknowledgment. [Kevin]
- Chore: updated readme and contribution guide. [Kevin]
- Chore: basic framework is ready. [Kevin]
- Chore: developing input validation functions. [Kevin]
- Chore: linting. [Kevin]
- Chore: added issue and PR templates. [Kevin]
- Chore: added issue and PR templates. [Kevin]
- Chore: initial commit. [Kevin]

Docs
~~~~
- Docs: adding description of tutorials. [Kevin M. Jablonka]
- Docs: added some links to API docs, explain which class to use. Closes
  #78 (#80) [Kevin Jablonka]
- Docs: moving notes about class implementation to developer notes.
  [Kevin M. Jablonka]
- Docs: adding screenshots of tutorials that can be linked to mybinder.
  [Kevin M. Jablonka]
- Docs: pypal -> PyPAL. [Kevin M. Jablonka]
- Docs: rebuild docs. [Kevin M. Jablonka]
- Docs: pypal -> PyPAL in text. [Kevin M. Jablonka]
- Docs: added note about coef_var_threshold, closes #71. [Kevin M.
  Jablonka]
- Docs: citation placeholder added (#70) [Kevin Jablonka]

  * fix: warning message for mae_variance comparison

  * chore: added citation placeholder
- Docs: move beta to background. [Kevin M. Jablonka]
- Docs: added some first dicussion about the hyperparameters. [Kevin M.
  Jablonka]
- Docs: added some first dicussion about the hyperparameters. [Kevin M.
  Jablonka]
- Docs: fix typo in the list of attributes/properties. [Kevin M.
  Jablonka]
- Docs: adding some property docs (#57) [Kevin Jablonka]
- Docs: fix typo in docs. [Kevin M. Jablonka]
- Docs: fixing some typos, addings some notes about plotting and
  plotting api docs, #29. [Kevin M. Jablonka]
- Docs: updating hints about the crossvalidation. [Kevin M. Jablonka]
- Docs: updating hints about the crossvalidation. [Kevin M. Jablonka]
- Docs: updating hints about the crossvalidation. [Kevin M. Jablonka]
- Docs: added some hints about GPR, closes #44 (#46) [Kevin Jablonka]
- Docs: fixed typo. [Kevin M. Jablonka]
- Docs: fixed typo. [Kevin M. Jablonka]
- Docs: inline code in sphinx docs. [Kevin M. Jablonka]
- Docs: fix some typos in readme, rebuilt docs. [Kevin M. Jablonka]
- Docs: fix some typos in readme, rebuilt docs. [Kevin M. Jablonka]
- Docs: added docstring to the PAL classes #40 (#43) [Kevin Jablonka]
- Docs: updating notes on beta. [Kevin M. Jablonka]
- Docs: adding beta influence. [Kevin M. Jablonka]
- Docs: adding beta influence. [Kevin M. Jablonka]
- Docs: adding beta influence. [Kevin M. Jablonka]

Fix
~~~
- Warning message for mae_variance comparison. [Kevin M. Jablonka]
- Crossvalidation returned only nan due to wrong if. [Kevin M. Jablonka]
- Replace nan MAE by inf. [Kevin M. Jablonka]
- Indices in test fixed. [Kevin M. Jablonka]
- Start iteration count at 1. [Kevin M. Jablonka]
- Fixes remaining typos for uncertainity. [byooooo]
- Took two times sqrt in coregionalized pal. [Kevin]
- Training function for PALSklearn fixed. [Kevin]
- Coverage command in workflow was broken. [Kevin]
- Pareto_classify did not for as expected #4. [Kevin]
- Need GPy for the Pythonpackage workflow. [Kevin]
- Omit for report of coverage. [Kevin]
- Uncertainity region test no longer failing. [Kevin]
- Should also work with 3.6. [Kevin]
- Should also work with 3.6. [Kevin]
- Should also work with 3.8. [Kevin]
- Install package for python package workflow. [Kevin]
- Activating Python Package CI. [Kevin]
- Export SKIP env variable in the pre-commit step. [Kevin]
- Installing pylint for pre-commit CI workflow. [Kevin]

Other
~~~~~
- Update docs. [byooooo]
- Merge branch 'master' of github.com:kjappelbaum/PyPAL. [Kevin M.
  Jablonka]
- Merge branch 'master' of github.com:kjappelbaum/PyPAL. [Kevin M.
  Jablonka]
- Validate sklearn GaussianProcessRegressor and extract model from
  fitted GridSearchCV/RandomizedSearchCV (#69) [Kevin Jablonka]

  * fix: warning message for mae_variance comparison

  * feat: first implementation of sklearn gpr validation

  * feat: using new validation in PALSklearn

  * chore: updating docstring of PALsklearn

  * docs: rebuilding docs
- Docs spellcheck (#63) [Kevin Jablonka]

  * chore: spellcheck on landing page

  * chore: updating developer notes

  * docs: some spellchecking of the docs
- Merge branch 'master' of github.com:kjappelbaum/PyPAL. [Kevin M.
  Jablonka]
- Merge pull request #31 from kjappelbaum/docs. [Kevin Jablonka]

  Docs
- Add prospector, closes #2. [Kevin M. Jablonka]
- Add prospector, closes #2. [Kevin M. Jablonka]
- Add prospector, closes #2. [Kevin M. Jablonka]
- Add prospector, closes #2. [Kevin M. Jablonka]
- Merge pull request #21 from kjappelbaum/noise_kernel. [Kevin Jablonka]

  Now, using the mu and the std of the measurement
- Gitter added, closes #10. [Kevin]
