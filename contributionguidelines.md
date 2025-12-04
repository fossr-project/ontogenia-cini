# Bench4KE Contribution Guidelines

Welcome to **Bench4KE**!

Thank you for your interest in contributing. Whether you are a developer, researcher, or practitioner in Knowledge Engineering, your contributions are highly valued. Before you start, please take a moment to read these guidelines to help keep contributions consistent and easy to review.

---

## 1. How to Contribute?

| Contribution Type                          | How?                                                                                                                                                                                                                                                                                                                                                                           |
|:-------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1. Reporting Bugs                          | If you encounter a bug, crash, or unexpected behavior, please open an issue in the GitHub **Issue Tracker** with a clear title and description, steps to reproduce, expected vs. actual behavior, logs or screenshots if relevant, and information about your environment (OS, Python version, Bench4KE version/commit).                                                        |
| 2. Requesting Features or New Tasks        | If you’d like Bench4KE to support a new KE task (e.g., new CQ generation setting, CQ-to-SPARQL mapping, ontology alignment, etc.) or a new feature in the validator/UI, open an issue describing the motivation, desired behavior, a rough API or UI suggestion (if any), and any related work/papers.                                                                         |
| 3. Adding a New Dataset / Benchmark        | If you wish to add a new dataset (CQ set, ontology, user stories, etc.), open an issue describing the domain, size, license, format, and how it fits into Bench4KE’s overall scope. Include links to source materials and any preprocessing scripts if available. We will discuss how it can be integrated and how metadata should be provided.                                   |
| 4. Adding a New Metric or Evaluation Setup | If you propose a new metric family (e.g., semantic similarity, coverage, LLM-as-a-judge variant), open an issue explaining the metric, the rationale, any dependencies required, and example inputs/outputs. If you already have a prototype, you can directly submit a pull request and link it from the issue.                                                                |
| 5. Integrating a New System for Evaluation | If you maintain or built a CQ generation system (or another KE automation system) that should be evaluated via Bench4KE, open an issue describing how it can be connected (REST API, CLI, Python library, etc.) and provide example commands or endpoints. Optionally, propose a small integration script as a starting point via pull request.                                    |
| 6. Improving Documentation                 | If you find parts of the documentation unclear, missing, or outdated, feel free to open an issue with improvement suggestions or directly modify the docs (README, tutorials, usage examples) and submit a pull request.                                                                                                                                                       |
| 7. Improving Infrastructure & Tooling      | If you want to improve the CI, tests, code quality, or developer experience (e.g., Docker setup, pre-commit hooks, linters), please outline your proposal in an issue first so we can agree on the general direction before you invest a lot of time.                                                                                                                          |

---

## 2. Commit Guidelines

To keep the history readable and reviews efficient, please follow these commit guidelines:

- **One commit per logical change**  
  Try to keep each commit focused. Avoid mixing unrelated changes (e.g., refactoring and feature additions) in a single commit.

- **Keep changes small and coherent**  
  Smaller, self-contained commits are easier to review, test, and revert if necessary.

- **Separate formatting from logic**  
  Don’t mix pure formatting changes (e.g., running a formatter or linter) with functional updates whenever possible.

- **Write clear commit messages**  
  - Use **imperative mood**, e.g., `Add CQ overlap metric`, `Fix validator heatmap generation`, `Update dataset metadata`.
  - Keep the subject line concise (ideally ≤ 60 characters).
  - Focus on what the change does, not how it is implemented.
  - Reference issues/PRs as needed, e.g., `Add CQ overlap metric [#12]`.

- **Optional emojis in commit messages**

You may (optionally) use emoji codes for additional clarity:

| Code                   | Emoji | Use for                           |
|------------------------|-------|-----------------------------------|
| `:fire:`               | 🔥    | Remove code or files              |
| `:bug:`                | 🐛    | Fix a bug or issue                |
| `:sparkles:`           | ✨    | Add feature or improvement        |
| `:memo:`               | 📝    | Add or update documentation       |
| `:tada:`               | 🎉    | Project or major feature start    |
| `:recycle:`            | ♻️    | Refactor code                     |
| `:pencil2:`            | ✏️    | Minor changes or small fixes      |
| `:bookmark:`           | 🔖    | Tag or version release-related    |
| `:adhesive_bandage:`   | 🩹    | Non-critical fix                  |
| `:test_tube:`          | 🧪    | Tests and CI-related changes      |
| `:boom:`               | 💥    | Breaking changes                  |

Use them if you enjoy them—otherwise, plain text is perfectly fine.


