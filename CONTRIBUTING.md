# Contributing

Bug reports and fixes are welcome.

* **Bugs:** open an issue with the bug report form. The Doctor report (*Setup / Doctor*,
  Doctor tab) and the end of `plugin.log` answer most first questions;
  [docs/INSTALL.md](docs/INSTALL.md#where-the-logs-are) says where the log is.
* **Small fixes** (a typo, an obvious bug with a test): open a pull request directly.
* **Anything larger** (a new feature, a change to the daemon's HTTP contract in
  [API.md](plugin/sam3_gimp/_daemon/API.md), a new dependency): open an issue first, so the
  approach is agreed before you spend time on it.

A pull request needs:

* the test suite passing: `python3 -m pytest`. No GIMP, GPU or torch is needed; see
  [docs/DEVELOPING.md](docs/DEVELOPING.md) for setup.
* for a bug fix, a test that fails without the change.
* code that reads like the code around it.
* no third-party imports on the plug-in side. It runs on GIMP's own Python with the
  standard library only.

Security problems go through private reporting, not issues or pull requests: see
[SECURITY.md](SECURITY.md).

Contributions are licensed under the MIT License, the same as the project.
