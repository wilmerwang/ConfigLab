from configlab.utils.git_utils import (
    _git,
    get_branch,
    get_commit,
    get_diff_staged,
    get_diff_tracked,
    get_untracked_files,
)


def test_git_utils() -> None:
    """Test the git utilities."""
    # Test get_commit
    commit = get_commit()
    assert isinstance(commit, str)
    assert len(commit) == 40

    # Test get_branch
    branch = get_branch()
    assert isinstance(branch, str)
    assert len(branch) > 0

    # Test get_diff
    diff = get_diff_tracked()
    assert isinstance(diff, str)

    # Test get_diff_staged
    diff_staged = get_diff_staged()
    assert isinstance(diff_staged, str)

    # Test get_untracked_files
    untracked_files = get_untracked_files()
    assert isinstance(untracked_files, list)
    for file in untracked_files:
        assert isinstance(file, str)

    # Test _git with an allowed command
    status = _git(["status"])
    assert isinstance(status, str)

    # Test _git with a disallowed command
    try:
        _git(["invalid-command"])
        raise AssertionError("Expected ValueError for unsupported git command")
    except ValueError:
        pass
