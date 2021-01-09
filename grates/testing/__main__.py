# Copyright (c) 2020-2021 Andreas Kvas
# See LICENSE for copyright/license details.

import pytest
import argparse
import sys
import importlib.resources
from .gravityfield import test_cases as gravityfield_test_cases
from .utilities import test_cases as utilities_test_cases
from .grid import test_cases as grid_test_cases


def generate_data(test_cases):

    for test in test_cases:
        test.generate_data()


def delete_data(test_cases):

    for test in test_cases:
        test.delete_data()


def main(args):

    parser = argparse.ArgumentParser(description='Run grates tests')
    parser.add_argument('--run-tests', dest='run_tests', action='store_true', default=False,
                        help='Run all tests using pytest')
    parser.add_argument('--generate-data', dest='generate_data', action='store_true', default=False,
                        help='Generate data test for regression tests')
    parser.add_argument('--delete-data', dest='delete_data', action='store_true', default=False,
                        help='Delete data test for regression tests')

    args = parser.parse_args()

    test_cases = []
    test_cases.extend(gravityfield_test_cases)
    test_cases.extend(utilities_test_cases)
    test_cases.extend(grid_test_cases)

    if args.generate_data:
        generate_data(test_cases)
    elif args.delete_data:
        delete_data(test_cases)

    if args.run_tests:

        for file_name in ['kernel.py', 'utilities.py', 'gravityfield.py', 'grid.py']:
            with importlib.resources.path('grates.testing', file_name) as f:
                pytest.main([str(f)])


if __name__ == "__main__":
    main(sys.argv)
