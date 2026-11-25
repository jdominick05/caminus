# Security policy

## Report a vulnerability

Please report security problems **privately**, never in a public issue, pull request or discussion:

- Open a private report from the repository's
  [Security tab](https://github.com/jdominick05/caminus/security/advisories/new) ("Report a vulnerability").

Include what you can: the Caminus version (`caminus --version`) or commit, the command or setup, what an attacker can
do, and the steps or a proof of concept. Leave out private conversations, keys and code that is not needed to show
the problem.

Caminus is maintained by one person. Reports are acknowledged within 7 days, and you will hear what happens next:
a fix and a release, or why the report is out of scope. Please give us a reasonable time to ship a fix before you
publish; we credit reporters in the advisory and the changelog unless you would rather not be named.

## Supported versions

| Version | Security fixes |
|---|---|
| The latest release (0.3.x) | Yes |
| Nightly builds and the `testing` branch | Fixed there first, no advisory of their own |
| Older releases | No: upgrade to the latest release |

## Scope

In scope: the code in this repository and the packages built from it.

- **The server** (`caminus serve` and the CLI that starts it): for example, a request that reads or writes files, runs
  code or crashes the machine beyond the request itself.
- **Ember and resident agents**, measured against [Ember's security model](docs/ember.md#the-security-model): for
  example, the agent reaching your account or files, an API key leaving its relay, a root command running without
  your approval, or a `sandbox` / `browser` box reaching what [the tool reference](docs/resident-agent.md#tools) says
  it cannot.
- **The panel** and the local scripts the packages install.
- **Releases**: anything that breaks the provenance described in [verifying your download](docs/verifying.md).

Out of scope, because the docs already say these are not protections:

- **The server has no authentication.** It listens on `127.0.0.1` by default; serving on another address with
  `--host` gives everyone who can reach that address full use of it. Put an authenticating proxy in front if you need
  that.
- **The agent account is not a sandbox.** It can read world-readable files, use the network and, unless you remove it
  from the GPU groups, use the GPUs ([the security model](docs/ember.md#the-security-model)).
- **What a model writes.** A model can be talked into bad advice or a bad tool call; that is a report only when it
  crosses one of the boundaries above.
- **Dependencies and model weights** (PyTorch, ROCm, Hugging Face files): report those to their own projects. What
  Caminus checks about them is in [verifying your download](docs/verifying.md#what-this-does-not-cover).
