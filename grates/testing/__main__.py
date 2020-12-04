import pytest
import argparse
import sys
import importlib.resources


def main(args):

    parser = argparse.ArgumentParser(description='Run grates tests')
    parser.add_argument('--run-tests', dest='run_tests', action='store_true', default=False,
                        help='Run all tests using pytest')
    parser.add_argument('--generate-data', dest='generate_data', action='store_true', default=False,
                        help='Generate data test for regression tests')

    args = parser.parse_args()

    if args.generate_data:
        print('generate data')

    if args.run_tests:

        for file_name in ['kernel.py', 'utilities.py']:
            with importlib.resources.path('grates.testing', file_name) as f:
                pytest.main([str(f)])


if __name__ == "__main__":
    main(sys.argv)
