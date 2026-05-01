import json
import shutil
import subprocess
from pathlib import Path

ALLOWED_GIT_COMMANDS = {
    "status",
    "rev-parse",
    "log",
    "diff",
    "ls-files",
}


def _git(cmd: list[str], cwd: Path = Path(".")) -> str:
    if not cmd or cmd[0] not in ALLOWED_GIT_COMMANDS:
        raise ValueError("Unsupported git command")

    result = subprocess.run(  # noqa: S603
        ["git", *cmd],  # noqa: S607
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def get_commit(cwd: Path = Path(".")) -> str:
    """Get the current git commit hash."""
    return _git(["rev-parse", "HEAD"], cwd=cwd)


def get_branch(cwd: Path = Path(".")) -> str:
    """Get the current git branch."""
    return _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd)


def get_diff_tracked(cwd: Path = Path(".")) -> str:
    """Get the git diff."""
    return _git(["diff"], cwd=cwd)


def get_diff_staged(cwd: Path = Path(".")) -> str:
    """Get the git diff for staged changes."""
    return _git(["diff", "--staged"], cwd=cwd)


def get_untracked_files(cwd: Path = Path(".")) -> list[str]:
    """Get the list of untracked files."""
    return _git(["ls-files", "--others", "--exclude-standard"], cwd=cwd).splitlines()


def capture_git_info(cwd: Path = Path(".")) -> dict[str, str | list[str]]:
    """Capture git information."""
    return {
        "commit": get_commit(cwd=cwd),
        "branch": get_branch(cwd=cwd),
        "diff_tracked": get_diff_tracked(cwd=cwd),
        "diff_staged": get_diff_staged(cwd=cwd),
        "untracked_files": get_untracked_files(cwd=cwd),
    }


def snapshot_git_state(output_dir: Path, cwd: Path = Path(".")) -> None:
    """Snapshot the git state to a file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    git_info = capture_git_info(cwd=cwd)
    output_file = output_dir / "git_snapshot.json"
    with output_file.open("w") as f:
        json.dump(git_info, f, indent=4)

    for file in git_info["untracked_files"]:
        file_path = cwd / file
        if file_path.exists():
            dest_path = output_dir / file
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, dest_path)

    (output_dir / "diff_tracked.patch").write_text(git_info["diff_tracked"])
