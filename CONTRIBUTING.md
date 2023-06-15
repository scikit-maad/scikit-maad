# Contributors Guide

Whether you are a novice or experienced software developer, all contributions and suggestions are welcome!
The main ways to contribute are:

- [Reporting bugs](#reporting-bugs)
- [Present a solution to a reported issue](#solution-to-issue)
- [Add a tutorial to the example gallery](#add-tutorial-to-the-example-gallery)
- [Improve the documentation](#improve-documentation)
- [Develop new features](#develop-new-features)

Please note that any contributions you make will be under the BSD-3-Clause software license.

## Development installation

Clone repository and symlink the module into site-packages with `flit`.
```
git clone https://github.com/scikit-maad/scikit-maad.git
cd ~/scikit-maad
flit install --symlink
```

## Reporting bugs

Bugs are reported using the bug template, so if you find a bug create a new issue [here](https://github.com/scikit-maad/scikit-maad/issues/new?assignees=&labels=bug&template=bug_report.md&title=BUG%3A+). Please try to fill out the template with as much detail as you can.

## Solution to issue

If you know the solution to an issue, you can give feedback on the issues list and submit a fix through a pull request. See section [Develop new features](#develop-new-features) for a detailed step by step to submit a pull request.

## Add tutorial to the example gallery

Tutorials are a great way to show the functionalities of scikit-maad. If you have ideas, please create an issue using the [feature request template](https://github.com/scikit-maad/scikit-maad/issues/new?assignees=&labels=enhancement&template=feature_request.md&title=ENH%3A+).

## Improve documentation

If you’re browsing the documentation and notice a typo or something that could be improved, please consider letting us know by creating an issue or submitting a fix.

## Develop new Features
Improvements and new features are greatly appreciated. If you would like to contribute developing new features or making improvements to the available package, please follow the following steps:

1. Create an issue at [github](https://github.com/scikit-maad/scikit-maad/issues) describing the feature or bug.
2. The development team will determine if the feature should be added, or the bug fixed.
3. Discuss implementation details on the issue thread and reach consensus.
4. Once consensus is reached, start a pull request. Be sure to **always discuss the feature/fix before creating a pull request**.
5. Implement the feature/fix ensuring all test unit pass and full documentation is available and updated.
6. Request code review once the pull request is ready for review.
7. Fix requested changes until the pull requested is approved.
8. Once approved, the changes will be merged into the development branch. If necessary, the package will have a new release.

## Development standards
In order to make your contribution best suited and keep a homogeneous package, we ask to follow some basic standards.

### Documentation

Reliable documentation is a must. It will make information easily accessible, helping new users learn quickly. For scikit-maad, we ask to follow the [numpy format](https://numpydoc.readthedocs.io/en/latest/example.html#). This allows us to have compile an online html documentation with [Sphynx](https://www.sphinx-doc.org/en/master/).

You can test the docstring format with:
```

python -m doctest -v filename.py
```
### Standard variable names

When developing new functionalities, please use the consistent variable names:
- audio signal: `s`
- Frequency sampling or sampling frequency: `fs`
- Spectrogram: `Sxx`
- Regions of interest: should be a pandas DataFrame with columns `min_t`, `max_t`, `min_f`, `max_f`
