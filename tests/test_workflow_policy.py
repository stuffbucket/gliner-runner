from __future__ import annotations

from scripts.check_workflow_policy import check, check_text

SHA = "3d3c42e5aac5ba805825da76410c181273ba90b1"


def workflow(job: str) -> str:
    return f"name: T\non: push\njobs:\n{job}"


def test_repository_workflows_follow_the_policy() -> None:
    assert check() == []


def test_a_pinned_action_with_a_version_comment_passes() -> None:
    text = workflow(
        "  build:\n"
        "    runs-on: ubuntu-24.04\n"
        "    timeout-minutes: 5\n"
        "    steps:\n"
        f"      - uses: actions/checkout@{SHA} # v7.0.1\n"
        f"      - uses: Homebrew/actions/setup-homebrew@{SHA} # 2026.10.26.1\n"
    )
    assert check_text("t.yml", text) == []


def test_a_tag_or_short_sha_is_not_a_pin() -> None:
    text = workflow(
        "  build:\n"
        "    runs-on: ubuntu-24.04\n"
        "    timeout-minutes: 5\n"
        "    steps:\n"
        "      - uses: actions/checkout@v7\n"
        "      - uses: actions/checkout@3d3c42e # v7.0.1\n"
    )
    errors = check_text("t.yml", text)
    assert len(errors) == 2
    assert all("not pinned to a full commit SHA" in error for error in errors)


def test_a_pin_needs_a_version_not_a_branch_name() -> None:
    text = workflow(
        "  build:\n"
        "    runs-on: ubuntu-24.04\n"
        "    timeout-minutes: 5\n"
        "    steps:\n"
        f"      - uses: Homebrew/actions/setup-homebrew@{SHA} # main\n"
        f"      - uses: actions/checkout@{SHA}\n"
    )
    errors = check_text("t.yml", text)
    assert len(errors) == 2
    assert all("needs a version comment" in error for error in errors)


def test_every_job_that_runs_steps_has_a_timeout() -> None:
    text = workflow(
        "  first:\n"
        "    runs-on: ubuntu-24.04\n"
        "    steps:\n"
        "      - run: echo\n"
        "  second:\n"
        "    runs-on: ubuntu-24.04\n"
        "    timeout-minutes: 5\n"
        "    steps:\n"
        "      - run: echo\n"
        "  last:\n"
        "    runs-on: ubuntu-24.04\n"
        "    steps:\n"
        "      - run: echo\n"
    )
    assert check_text("t.yml", text) == [
        "t.yml:4: job first has no timeout-minutes",
        "t.yml:13: job last has no timeout-minutes",
    ]


def test_a_job_that_calls_a_reusable_workflow_needs_no_timeout() -> None:
    text = workflow(
        "  build:\n"
        "    uses: ./.github/workflows/_release-build.yml\n"
        "    with:\n"
        "      ref: main\n"
    )
    assert check_text("t.yml", text) == []
