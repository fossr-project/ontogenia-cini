# Bench4KE Maintenance Plan

Bench4KE is an extensible benchmarking system for Knowledge Engineering (KE) automation, currently focused on evaluating automatic Competency Question (CQ) generation. It exposes an API-centric architecture, ships with a curated multi-domain gold-standard dataset, and supports a variety of similarity and LLM-based evaluation metrics.  

This Maintenance Plan describes how Bench4KE will be kept reliable, extensible, and useful for the community over time.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17712424.svg)](https://doi.org/10.5281/zenodo.17712424)

---

## 1. Objectives & Responsibilities

### 1.1 Maintenance Objectives

The main objectives for Bench4KE maintenance are to:

- **Ensure long-term availability and stability**
  - Keep the benchmark API and dataset accessible.
  - Fix bugs and regressions in the orchestrator, validator, and front-end.
- **Support reproducible, comparable evaluation**
  - Maintain stable benchmark configurations and default datasets.
  - Provide versioned releases of code, datasets, and experiment setups.
- **Extend coverage of KE automation tasks**
  - Gradually support new tasks (e.g., CQ-to-SPARQL, ontology drafting, ontology testing, requirements extraction, ontology alignment).
- **Evolve metrics and datasets**
  - Add new similarity, coverage, and LLM-as-a-judge metrics.
  - Extend the gold-standard dataset with additional domains, formats, and usage scenarios while keeping quality high.
- **Maintain interoperability and standards compliance**
  - Preserve the REST API contract and JSON/CSV outputs.
  - Keep JSON-LD and DCAT-AP metadata in sync with releases.
- **Foster community-driven development**
  - Make it easy to contribute datasets, integrations, and features.
  - Use transparent processes for issues, pull requests, and discussions.

### 1.2 Roles & Responsibilities

The following roles define how maintenance activities are shared. Concrete people in each role may evolve over time and will be listed in the project README and `CONTRIBUTING.md`.

- **Lead Maintainers**
  - Own the overall technical direction of Bench4KE.
  - Approve major architectural changes and new benchmark tasks.
  - Coordinate release planning and ensure alignment with the project vision.

- **Core Developers**
  - Implement new features, metrics, and integrations.
  - Maintain the benchmark datasets, orchestrator, validator, and UI.
  - Review and merge pull requests from external contributors.

- **Data & Benchmark Stewards**
  - Curate and validate new datasets and CQs before they become part of the official benchmark.
  - Ensure that submitted datasets meet quality and licensing requirements.
  - Monitor dataset balance across domains and usage scenarios.

- **Infrastructure & DevOps Contributors**
  - Maintain CI configurations, test suites, and automation scripts.
  - Keep dependencies and runtime environments up to date.
  - Ensure reproducible environment setup (Docker images, requirements files, etc.).

- **Community Contributors**
  - Propose new metrics or evaluation strategies.
  - Integrate third-party CQ generation systems with Bench4KE.
  - Report issues, suggest enhancements, and participate in discussions.

---

## 2. Current Ongoing Maintenance and Plans

The following roadmap summarizes ongoing and planned activities for Bench4KE.  
Status codes: **Done**, **Ongoing**, **To-Do**.

| Category          | Description                                                                                                                                                                                                                                                                                   | Status   |
|:------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------:|
| Core Benchmark    | Maintain the default gold-standard dataset (17 ontology-driven projects, 843 CQs) and associated configuration files.                                                                                                                                                                          | Ongoing  |
| Datasets          | Add new domains, projects, and usage scenarios (user stories, ontologies, datasets, PDFs) through a curated contribution process, keeping balance across domains and formats.                                                                                                                 | To-Do    |
| Dataset Formats   | Extend support beyond the current XML/JSON/CSV-centric focus to additional structured and semi-structured formats where relevant.                                                                                                                                                             | To-Do    |
| Metrics           | Maintain existing metrics (Cosine/sBERT, BERTScore, Hit Rate, Jaccard, BLEU, ROUGE-L, LLM-as-a-judge) and add new measures that better capture semantic novelty and ontology coverage.                                                                                                        | Ongoing  |
| LLM Backends      | Keep compatibility with supported providers (e.g., OpenAI, Anthropic, Meta, Together AI) and allow configuration of model versions and parameters from a single settings file/environment configuration.                                                                                      | Ongoing  |
| API & Orchestrator| Stabilize and document the `/validate` endpoint and orchestrator flow (dataset selection, external service calls, metric computation), ensuring backward-compatible changes whenever possible.                                                                                                | Ongoing  |
| CQ Validator      | Maintain and extend the validator service (heatmap generation, CSV export, JSON output structure); optimize performance for larger datasets and incremental runs.                                                                                                                             | Ongoing  |
| Front-end UI      | Maintain the lightweight local web UI for running evaluations, showing metric summaries, and linking to stored outputs. Extend it with improved usability and basic analytics dashboards where feasible.                                                                                       | O-Going  |
| Extensible APIs   | Generalize the architecture so the same pipeline can support additional KE tasks (e.g., CQ-to-SPARQL mapping, ontology alignment, ontology generation/testing) via separate endpoints and task-specific validators.                                                                          | To-Do    |
| Data Leakage Mitigation | Provide guidance and configuration options for running evaluations in offline/controlled environments and for mitigating data leakage risks when using LLMs (e.g., local models, restricted connectivity, synthetic benchmarks).                                                            | To-Do    |
| Testing & CI      | Add and maintain unit tests and integration tests for the orchestrator, validator, and UI; set up CI workflows (linting, tests, build) for every PR and tagged release.                                                                                                                       | To-Do    |
| Documentation     | Keep installation, configuration, and integration guides up to date; provide examples for integrating CQ generation services and adding new datasets/metrics.                                                                                                                                 | Ongoing  |
| Community & Challenges | Use Bench4KE as the backbone for community evaluations, tutorials, courses, and shared challenges around KE automation; publish baseline results and encourage “Evaluated with Bench4KE” badges in downstream projects.                                                                  | Ongoing  |
| Governance        | Clarify acceptance criteria for new tasks, datasets, and metrics; document the decision process for promoting contributions from “experimental” to “official” status within the benchmark.                                                                                                    | To-Do    |

> **Contributions of new datasets, tasks, or metrics are welcome via GitHub Issues and Pull Requests.** Clear templates for each contribution type are provided in `CONTRIBUTING.md`.

---

## 3. Maintenance Regulations

### 3.1 Code Quality

- Enforce a consistent coding style (PEP 8 for Python-based components).
- Prefer small, focused pull requests with clear descriptions and tests.
- Refactor legacy or experimental modules as the system evolves, reducing technical debt while preserving documented public interfaces.

### 3.2 Version Control & Releases

- Use **Git** as the canonical version control system, hosted on GitHub.
- Follow **semantic versioning** for Bench4KE releases:
  - **MAJOR**: Breaking changes in APIs, data formats, or evaluation semantics.
  - **MINOR**: Backwards-compatible additions (new metrics, datasets, tasks).
  - **PATCH**: Bug fixes, documentation updates, minor corrections.
- Tag every release in Git and keep the repository synchronized with Zenodo for DOI assignment.
- Keep a detailed **CHANGELOG.md** describing:
  - Added / Changed / Deprecated / Fixed entries.
  - Any migration steps required for users (e.g., configuration changes).

### 3.3 Release Cadence

Targeted (non-binding) cadence:

- **Major releases**: Every 6–12 months or when significant breaking changes accumulate.
- **Minor releases**: When substantial new functionality is completed (e.g., a new KE task or metric family).
- **Patch releases**: As needed, especially for critical bugs or dependency-related issues.

Release cadence may be adapted based on community needs and available maintainer time.

### 3.4 Documentation

- Maintain user-facing documentation in the repository (e.g. `docs/` and README):
  - Quickstart guides for:
    - Running the validator locally.
    - Integrating an external CQ generation service via the REST API.
    - Running evaluations via the front-end UI.
  - Task-specific guides (e.g., CQ generation now; more tasks later).
- Document environment setup:
  - Python version requirements.
  - Dependency management (e.g., `requirements.txt`, `pyproject.toml`, Dockerfiles).
- Keep **API contracts** documented (request/response schemas, parameters, expected formats).
- Provide “How to contribute” and “How to add a dataset/metric/task” guides.

### 3.5 Compatibility & Dependencies

- Regularly review dependencies for:
  - Security advisories.
  - Compatibility with supported Python versions.
- Maintain a minimal set of officially supported Python versions (e.g., `3.10+`), documented in README and CI.
- For LLM integrations:
  - Allow configuration of API keys and endpoints through environment variables or config files.
  - Avoid hard-coding provider-specific behavior where generic interfaces are possible.

### 3.6 Security & Data Handling

- **Never** commit secrets (API keys, access tokens) to the repository.
- Clearly document how to provide credentials (env vars, local config files ignored by Git).
- Provide guidelines for:
  - Running Bench4KE in environments with restricted network access.
  - Evaluating systems without exposing sensitive datasets to external LLM providers.
- Review contributions affecting data loading and remote calls with particular care.

---

## 4. Community Engagement

- Use **GitHub Issues** to track bugs, feature requests, and proposals for new datasets, metrics, or KE tasks.
  - Aim to acknowledge new issues within **7–10 days**.
- Use **Pull Requests** for code and dataset contributions.
  - Require at least one maintainer review before merging.
- Maintain a **Code of Conduct** to ensure a welcoming, inclusive environment for contributors.
- Encourage external projects that integrate with Bench4KE to:
  - Document their integration.
  - Optionally display an “Evaluated with Bench4KE” note/badge in their README.
- Collect feedback via:
  - Short surveys on usability and usefulness.
  - Discussions in issues and GitHub Discussions (if enabled).
- Use workshops, conference tutorials, and KE courses to:
  - Present Bench4KE.
  - Gather suggestions from practitioners and researchers.
  - Encourage community contributions to extend the benchmark.

---

## 5. Availability, Sustainability & Licensing

- **Repository & Archival**
  - Source code, datasets, experiment configs, and results are hosted on GitHub.
  - Each stable release is archived on **Zenodo** and referenced via DOI: `10.5281/zenodo.17712424`.
- **Licensing**
  - Bench4KE is released under the **Apache License 2.0**, enabling reuse, extension, and integration in other projects.
- **Sustainability Commitments**
  - Long-term maintenance is supported by the institutions behind the project and by the broader research community.
  - At least a 5-year support horizon is anticipated for the current codebase and dataset, with ongoing efforts to grow a distributed maintainer team.
- **Standards & Interoperability**
  - The REST API returns JSON and CSV (and JSON-LD where applicable).
  - Dataset metadata is described using **DCAT-AP**, improving findability and interoperability with open data portals.

---

## 6. Contact & Support

For questions about using or extending Bench4KE, or for proposals to add new datasets/metrics/tasks, please:

- Open an **Issue** on the GitHub repository (preferred).
- Submit a **Pull Request** for concrete contributions (code, configs, docs, datasets).
- Use project-maintainer contact information listed in the repository README if direct communication is necessary (e.g., for institutional collaborations or sensitive data discussions).

> Bench4KE is intended as a community benchmark. Contributions that improve its coverage, fairness, and robustness are strongly encouraged.
