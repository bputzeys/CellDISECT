============
Contributing
============

We love your input! We want to make contributing to CellDISECT as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

Development Process
--------------------

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

1. Fork the repo and create your branch from ``main``.
2. If you've changed APIs, update the documentation.
3. Make sure your code lints.
4. Issue that pull request!

Pull Request Process
---------------------

1. Update the README.md with details of changes to the interface, if applicable.
2. Update the docs/source/changelog.rst with a note describing your changes.
3. The PR will be merged once you have the sign-off of at least one other developer.

Any contributions you make will be under the BSD 3-Clause License
------------------------------------------------------------------

In short, when you submit code changes, your submissions are understood to be under the same `BSD 3-Clause License <https://opensource.org/licenses/BSD-3-Clause>`_ that covers the project. Feel free to contact the maintainers if that's a concern.

Report bugs using GitHub's `issue tracker <https://github.com/Lotfollahi-lab/CellDISECT/issues>`_
----------------------------------------------------------------------------------------------------------

We use GitHub issues to track public bugs. Report a bug by `opening a new issue <https://github.com/Lotfollahi-lab/CellDISECT/issues/new>`_; it's that easy!

Write bug reports with detail, background, and sample code
-----------------------------------------------------------

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can.
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

Development Setup
-------------------

Here's how to set up CellDISECT for local development:

1. Fork the CellDISECT repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/CellDISECT.git

3. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

4. Create a conda environment and install dependencies::

    $ conda create -n celldisect-dev python=3.9
    $ conda activate celldisect-dev
    $ pip install -e ".[dev]"

5. Make your changes locally.

6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website.

License
-------

By contributing, you agree that your contributions will be licensed under its BSD 3-Clause License. 