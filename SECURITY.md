# Security

Please report security problems privately, not in a public issue: use **Report a
vulnerability** on the repository's
[Security tab](https://github.com/JamesAntisdel/gimp-sam3-plugin/security/advisories/new).

A useful report says what an attacker could do, how to reproduce it, and which version you
ran. The line under the preview in the plug-in's window shows the daemon version and the
plug-in build.

In scope: the plug-in, the bundled daemon (`sam3gimpd`), Setup, and the installer scripts.
By default the daemon listens only on 127.0.0.1 and requires a per-run token. Reports about
how it authenticates, what it accepts, or what Setup downloads and runs are especially
welcome. SAM 3, PyTorch, transformers and GIMP are upstream projects; please report problems
in them to their own maintainers.

Reports are read by the maintainer; there is no fixed response time.
