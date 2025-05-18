***************
Contributing
***************

Thank you for your interest in contributing to pyplis! This guide will help you get started.

Git Branching Strategy
=====================

pyplis uses the following branching strategy:

- **master**: Contains only stable, released code. New commits only arrive here through merges from ``next-release`` when preparing a new release.
- **next-release**: The primary development branch where new features and fixes are integrated for the upcoming release.
- **feature branches**: Created from ``next-release`` for developing new features or fixes.

Workflow
--------

1. Create a feature branch from ``next-release``::

    git checkout next-release
    git checkout -b feature/your-feature-name

2. Make your changes, commit them, and push to your fork
3. Create a pull request targeting the ``next-release`` branch
4. After review and approval, your changes will be merged into ``next-release``
5. When ready for release, ``next-release`` will be merged into ``master`` and a new release tag created

This strategy ensures that:
- The master branch always contains stable, released code
- Issues remain open until features/fixes are properly released
- Development and testing can proceed without affecting the stable release
- The branch name clearly indicates where unreleased changes are staged

Making Changes
=============

1. Fork the repository on GitHub
2. Clone your fork locally::

    git clone https://github.com/your-username/pyplis.git
    cd pyplis
    git remote add upstream https://github.com/jgliss/pyplis.git

3. Create a feature branch as described above
4. Make your changes
5. Add tests if applicable
6. Update documentation if needed
7. Submit a pull request

For more detailed information about the codebase please refer to the :ref:`api` documentation.