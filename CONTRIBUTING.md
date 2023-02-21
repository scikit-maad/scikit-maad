# Contributors Guide

There are multiple ways to contribute to this collaborative project:

- Reporting issues.
- Present a solution to a reported issue.
- Add a tutorial to the example gallery showing how to use scikit-maad functionalities.
- Improve the API documentation.
- Develop new features.

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
### Variable names
- audio signal: `s`
- Frequency sampling or sampling frequency: `fs`
- Spectrogram: `Sxx`
- Regions of interest: should be a pandas DataFrame with columns `min_t`, `max_t`, `min_f`, `max_f`
