# Quizzer Config Init — Patch for Remaining Minor Issues

## Summary

The original feature implementation (feat/0012) was marked as done, but two minor
issues were identified during code review:

1. **`max_tokens` divergence in `ai_generate_mcqs_for_topic()`:** The function passes a
   hardcoded `800` to `_chat_completion_content()` instead of resolving from config,
   while the spec defines `max_tokens=600` as the default. By contrast,
   `ai_extract_topics()` correctly uses `max_tokens=600`.

2. **pytest mark warning:** `tests/test_quizzer_config.py` uses
   `pytestmark = pytest.mark.integration`, but the `integration` marker is not registered
   in `[tool.pytest.ini_options]` in `pyproject.toml`, producing a runtime warning.

## Fix for Issue 1 — max_tokens divergence

### Current state

In `src/study_utils/quizzer/manager/quiz.py`:

- `ai_extract_topics()` calls `_chat_completion_content(..., max_tokens=600)` at line 604,
  matching the spec default.
- `ai_generate_mcqs_for_topic()` calls `_chat_completion_content(..., max_tokens=800)` at
  line 482, using a hardcoded literal that does not respect config overrides.

The implementation of `ai_generate_mcqs_for_topic()` already resolves `model` and
`temperature` from config when they are at defaults (lines 463–471). The fix is to apply
the same pattern for `max_tokens`.

### Proposed change — simpler call-site approach

Change only the `_chat_completion_content()` call in `ai_generate_mcqs_for_topic()`.
Resolve `max_tokens` from config using a local variable, mirroring the existing model/temperature
resolution block. No changes to the function signature or any other callers.

**Existing code (lines 463–482):**

```python
use_config_defaults = model == "gpt-4o-mini" and temperature == 0.2
if use_config_defaults:
    try:
        ai_conf = load_config().ai
        model = model or ai_conf.model
        temperature = temperature or ai_conf.temperature
    except QuizzerConfigError:
        pass

# ... later ...

content = _chat_completion_content(
    resolved_client,
    model=model,
    system_prompt=sys_prompt,
    user_prompt=user_prompt,
    temperature=temperature,
    max_tokens=800,  # <-- hardcoded
)
```

**New code:**

```python
config_max_tokens: int = 800
use_config_defaults = (
    model == "gpt-4o-mini" and temperature == 0.2 and max_tokens == 800
)
if use_config_defaults:
    try:
        ai_conf = load_config().ai
        model = model or ai_conf.model
        temperature = temperature or ai_conf.temperature
        config_max_tokens = ai_conf.max_tokens
    except QuizzerConfigError:
        pass

# ... later ...

content = _chat_completion_content(
    resolved_client,
    model=model,
    system_prompt=sys_prompt,
    user_prompt=user_prompt,
    temperature=temperature,
    max_tokens=config_max_tokens,  # <-- now config-derived
)
```

### Why this approach

- **Minimal surface area:** Only one function affected. No signature changes, no new
  parameters for existing callers to update.
- **Consistent semantics:** Same "config when at defaults" logic already used for model and
  temperature. If a caller explicitly passes `max_tokens=500`, it bypasses config (check
  `max_tokens == 800`).
- **Safe default:** When there is no `[ai]` section or load fails gracefully, the
  fallback remains 800 (the existing behavior).

### Test cases to add to `tests/test_quizzer_config.py`

Add two new tests in an existing or new class:

| # | Test name | What it verifies |
|---|-----------|------------------|
| A | `test_ai_generate_mcqs_resolves_max_tokens_from_config` | Write config with `[ai]\nmax_tokens = 600`. Call the internal path that exercises `ai_generate_mcqs_for_topic()`. Confirm `config_max_tokens` resolves to 600 (via mocking `_chat_completion_content`). |
| B | `test_ai_generate_mcqs_hardcoded_max_tokens_bypasses_config` | Pass `max_tokens=500` explicitly. Confirm config override is ignored and 500 is used. |

## Fix for Issue 2 — pytest mark registration

### Current state

`tests/test_quizzer_config.py:11` declares:
```python
pytestmark = pytest.mark.integration
```

But `pyproject.toml`'s `[tool.pytest.ini_options]` section (lines 83–85) has no
registered markers. Pytest emits this warning:

```
tests/test_quizzer_config.py:11
  PytestUnknownMarkWarning: Unknown pytest.mark.integration - is this a marker?
```

### Proposed change

Add the `integration` marker to `[tool.pytest.ini_options]` in `pyproject.toml`.

**File:** `pyproject.toml`, append after line 85:

```toml
[tool.pytest.ini_options]
addopts = "-q -n auto --cov=study_utils --cov-report=term-missing --cov-report=xml --cov-fail-under=90 --maxfail=1"
testpaths = ["tests"]
markers = [
    "integration: marks tests as integration tests (deselect with '-m \"not integration\"')",
]
```

No code changes. Only configuration addition to `pyproject.toml`.

### Why this approach

- **Single line of config:** Minimal change, no new files or conftest modifications.
- **Delegation enabled:** Users can deselect all integration tests via `-m "not integration"`
  in future if needed.
- **Consistent pattern:** Other pytest markers used throughout the project (e.g., `unit`, `e2e`)
  follow this same declaration pattern.

## Implementation checklist

- [x] Modify `src/study_utils/quizzer/manager/quiz.py` — add `config_max_tokens` resolution in `ai_generate_mcqs_for_topic()` (lines 463–474)
- [x] Add tests A and B to `tests/test_quizzer_config.py` (`TestMaxTokensResolution`, lines 327–414)
- [x] Update `pyproject.toml` — register `integration` marker (lines 86–88)
- [x] Run full test suite — all tests pass, no warnings confirmed
- [x] Lint checks — Ruff passes on all modified files

## Definition of done

- [x] `ai_generate_mcqs_for_topic()` resolves `max_tokens` from `[ai]` section when at default (line 472: `config_max_tokens = ai_conf.max_tokens`)
- [x] Hardcoded `max_tokens=800` remains as the fallback when config is unavailable (line 463, lines 473–474)
- [x] Two new test cases cover resolved and bypassed max_tokens paths (`test_ai_generate_mcqs_resolves_max_tokens_from_config`, `test_ai_generate_mcqs_hardcoded_max_tokens_bypasses_config`)
- [x] No pytest warnings (`PytestUnknownMarkWarning`) in test output
- [x] All existing tests continue to pass (backward compat)
- [x] `ruff check` passes on all affected files
