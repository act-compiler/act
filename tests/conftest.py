"""The platform the end-to-end test simulates on"""


def pytest_addoption(parser):
    parser.addoption(
        '--platform',
        choices=['CPU', 'GPU'],
        default='CPU',
        help='simulate the oracle on the CPU or the GPU',
    )
