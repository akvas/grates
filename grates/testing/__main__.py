# Copyright (c) 2020 Andreas Kvas
# See LICENSE for copyright/license details.

import pytest
import argparse
import sys
import importlib.resources
from .gravityfield import test_classes as gravityfield_test_classes
from .utilities import test_classes as utilities_test_classes


def generate_data(test_classes):

    for test in test_classes:
        test.generate_data()


def delete_data(test_classes):

    for test in test_classes:
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

    test_classes = []
    test_classes.extend(gravityfield_test_classes)
    test_classes.extend(utilities_test_classes)

    if args.generate_data:
        generate_data(test_classes)
    elif args.delete_data:
        delete_data(test_classes)

    if args.run_tests:

        for file_name in ['kernel.py', 'utilities.py', 'gravityfield.py', 'grid.py']:
            with importlib.resources.path('grates.testing', file_name) as f:
                pytest.main([str(f)])


if __name__ == "__main__":
    main(sys.argv)
