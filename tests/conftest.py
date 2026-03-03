import os

os.environ["FLUENT_BASE_API_KEY"] = "mock_key_for_testing"

import pytest

@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    return os.environ.get("FLUENT_BASE_API_KEY")