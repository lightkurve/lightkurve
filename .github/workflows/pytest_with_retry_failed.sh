#!/bin/sh

# see: https://docs.pytest.org/en/latest/how-to/cache.html#usage

pytest "$@"
if [ $? -ne 0 ]; then
    echo
    echo "====== Some tests failed. Re-run failed tests ======"
    echo
    pytest --last-failed "$@"
fi
