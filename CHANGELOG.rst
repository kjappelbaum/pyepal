Changelog
=========


(unreleased)
------------

New
~~~
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

Fix
~~~
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
