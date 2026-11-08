# Contributing to Caminus

Thanks for helping. Caminus has been tested on one machine so far, so you can help a lot without writing any code.

## Ways to help

- **Send a hardware report.** Run Caminus on your cards and
  [tell us how it went](https://github.com/jdominick05/caminus/issues/new?template=hardware_report.yml), even when it all
  worked. This is the most useful thing you can do right now.
- **Report a bug.** [Open an issue](https://github.com/jdominick05/caminus/issues/new?template=bug_report.yml) with the
  output of `caminus doctor`. Please leave out private conversations and code.
- **Ask for something.** A model, a card, a tool:
  [feature requests](https://github.com/jdominick05/caminus/issues/new?template=feature_request.yml) are welcome.
- **Improve the docs.** If a page confused you, a fix or an issue saying where you got stuck helps the next person.
- **Send code.** Read on.

## Sending a pull request

1. Fork the repository and make a branch for your change.
2. Set up a checkout: [building from source](docs/dev/building.md).
3. Make the change. Keep it focused: one change per pull request.
4. Run the CPU tests (no GPU needed): `python -I tests/ci_cpu.py --scratch <a fresh directory>`. CI runs the same.
5. If your change touches GPU code, run the GPU tests that cover it, following [GPU safety](docs/dev/gpu-safety.md), and
   say in the pull request which cards you ran them on.
6. Open the pull request **against the `testing` branch**, not `main`. Changes collect in `testing` and move to `main` for
   a release.

Commit messages look like `area : what changed`, for example `prefill : one launch checker per module`. Explain why in
the body, with the numbers you measured if it's about speed.

### The contributor licence agreement

Before your first pull request is merged, you'll be asked to agree to the
[contributor licence agreement](CLA.md) (CLA). You keep the copyright to your code; the CLA gives the maintainer a licence
to it that includes relicensing the project later, for example to offer a commercial licence next to the AGPL. Your code
stays under the AGPL in this repository either way.

Signing takes one comment: on your first pull request a bot posts a link to the agreement and the sentence to reply
with. One signature covers all your later pull requests.

### AI-assisted contributions

You're welcome to use AI coding tools. Add an `Assisted-by: <tool or model>` line to the commits it helped with, review
everything it wrote, and make sure the CPU tests pass. You're responsible for the change either way.

## Where to start reading

- [How it works](docs/dev/architecture.md): the engine, the prompt path, the cache and the server in one page.
- [Testing](docs/dev/testing.md): the CPU suites, the GPU tests and the accuracy gates.
- [Adding a model](docs/dev/adding-a-model.md): what a new model family needs.
- [GLOSSARY.md](GLOSSARY.md): the project's terms.
- [Design choices](docs/dev/design-choices.md): what's settled and why.
- [The design specs](docs/specs/): detailed contracts of the drafters, the prefill, the cache and the server.

Comments in the code cite labels from the private development history: decision numbers like D-281, step names like
`impl:package`, and plan documents that aren't in this repository. Treat them as labels; the code and the tests are the
authority.
