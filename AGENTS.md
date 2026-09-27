# AGENTS.md

Instructions and operating guidelines for Antigravity AI coding agents working on the **Ghaemieh OCR** project.

---

## 1. Core Mandates & Rules

1. **Version Tracking (`config.py` & Admin Panel / UI)**:
   - The project version starts at `2.0.0` (matching the current refactored OCR architecture).
   - The version must be maintained in [`Refactored_OCR/config.py`](file:///mnt/d/OCR/Refactored_OCR/config.py) as `VERSION = "x.y.z"`.
   - **Requirement**: Increment / update the version in [`Refactored_OCR/config.py`](file:///mnt/d/OCR/Refactored_OCR/config.py) with **every update or modification** made to the codebase.
   - **Admin Panel & UI Version Synchronization**: Whenever bumping `VERSION`, agents **MUST ALSO update the displayed version string and fallbacks** in the admin portal ([`Refactored_OCR/admin.html`](file:///mnt/d/OCR/Refactored_OCR/admin.html)) and UI templates ([`Refactored_OCR/index.html`](file:///mnt/d/OCR/Refactored_OCR/index.html), [`Refactored_OCR/login.html`](file:///mnt/d/OCR/Refactored_OCR/login.html)) so that the admin panel and public pages always reflect the exact current version.
   - Follow semantic versioning (`PATCH` for bug fixes/minor adjustments, `MINOR` for new features/endpoints, `MAJOR` for breaking architectural overhauls).

2. **Documentation & UI Integrity**:
   - Maintain docstrings, type hints, and comments across all Python modules.
   - Keep user-facing guides, API specs, and the RTL Persian web UI ([`index.html`](file:///mnt/d/OCR/Refactored_OCR/index.html), [`script.js`](file:///mnt/d/OCR/Refactored_OCR/script.js), [`style.css`](file:///mnt/d/OCR/Refactored_OCR/style.css)) synchronized with backend capabilities.
   - Preserve existing comments, docstrings, and RTL styling unless explicitly instructed.

3. **Commit Message Format**:
   - Every time changes are introduced, generate or use a commit message starting with the current version tag followed by conventional commit format:
     `V<x.y.z> - feat(or fix or anything): <description>`
   - Examples:
     - `V2.0.1 - fix: allow environment variable overrides in config.py`
     - `V2.1.0 - feat: implement streaming OCR progress for multi-page PDFs`
     - `V2.0.2 - chore: add AGENTS.md operating guidelines`

4. **Modular Model Architecture & Graceful Degradation**:
   - All OCR engines must strictly implement the [`BaseOCRModel`](file:///mnt/d/OCR/Refactored_OCR/models/base.py) abstract interface under [`Refactored_OCR/models/`](file:///mnt/d/OCR/Refactored_OCR/models/).
   - Heavy GPU models (`Qwen`, `Varco`) must gracefully degrade or handle lack of GPU / insufficient VRAM without crashing the entire service.
   - Image preprocessing and PDF manipulation logic must remain decoupled in [`Refactored_OCR/utils/`](file:///mnt/d/OCR/Refactored_OCR/utils/) (`image_processing.py`, `pdf_utils.py`).
   - LLM merging / post-processing must remain cleanly encapsulated in [`Refactored_OCR/services/merger.py`](file:///mnt/d/OCR/Refactored_OCR/services/merger.py).

5. **Environment & Configuration Priority**:
   - All configurations (ports, model toggles, API endpoints, credentials, thresholds) must be configurable via environment variables with safe defaults in [`config.py`](file:///mnt/d/OCR/Refactored_OCR/config.py).
   - Never hardcode external API endpoints, internal IP addresses, or secrets directly in service logic.

6. **WSL & Cross-Platform Compatibility**:
   - Development and execution take place in Linux / WSL environments.
   - Ensure file paths, Docker mountings, and line endings remain POSIX-compliant.

---

## 2. Project Overview

- **Purpose**: High-accuracy, multi-backend OCR and document extraction service (FastAPI + web UI) supporting Persian, Arabic, and English text extraction from images and PDFs, featuring an LLM-assisted dual-model correction pipeline.
- **Target Runtime**: Python 3.10+ on Linux / WSL / Docker.
- **Key Dependencies**: Standard async web frameworks (`FastAPI`, `Uvicorn`), `pytesseract`, `pdf2image`, `docling`, `opencv-python`, `Pillow`, `openai` (for OlmOCR and LLM merger), `pydantic`, `transformers` / `torch` (optional for GPU models).

---

## 3. Architecture & Code Conventions

- **Modular Design**:
  - `Refactored_OCR/config.py`: Application configuration, environment settings, and version metadata.
  - `Refactored_OCR/main.py`: FastAPI application lifespan, routing (`/ocr/image`, `/ocr/pdf`, `/health/models`), and static asset serving.
  - `Refactored_OCR/models/`: Model wrappers implementing `BaseOCRModel` (`tesseract.py`, `olm.py`, `docling.py`, `qwen.py`, `varco.py`).
  - `Refactored_OCR/services/`:
    - `ocr_service.py`: Pipeline coordinator for preprocessing, single/dual model execution, and LLM triggering.
    - `merger.py`: LLM-based text reconciliation and error-correction service.
  - `Refactored_OCR/utils/`: Independent helpers for image enhancements (`image_processing.py`) and PDF conversion (`pdf_utils.py`).
  - `Refactored_OCR/index.html`, `script.js`, `style.css`: RTL Persian user interface with side-by-side original/processed previews.
- **Code Style**:
  - PEP 8 compliant, explicit type annotations (`typing`), robust exception handling.
  - Non-blocking patterns and proper tempfile lifecycle management (cleanup temporary files in `finally` blocks).
  - Avoid hardcoded paths; use configuration and environment variables.

---

## 4. Agent Operating Procedures

When executing tasks:
1. **Analyze Context**: Inspect existing files and structure before adding new code.
2. **Implement Changes**: Ensure modular, testable, and clean code.
3. **Bump Version**: Update `VERSION` in [`Refactored_OCR/config.py`](file:///mnt/d/OCR/Refactored_OCR/config.py) as part of the changeset, and synchronize the version in the admin panel ([`Refactored_OCR/admin.html`](file:///mnt/d/OCR/Refactored_OCR/admin.html)) and user interface pages (`index.html`, `login.html`).
4. **Verify**: Run syntax checks, unit tests, or linting commands when applicable (`python3 -m py_compile ...`).
5. **Report & Commit Message**: Summarize changes concisely with clickable links to modified files, and provide the commit message in the format `V<x.y.z> - <type>: <description>`.
