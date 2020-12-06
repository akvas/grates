# Copyright (c) 2020 Andreas Kvas
# See LICENSE for copyright/license details.

import abc


class TestCase:

    @abc.abstractmethod
    def generate_data(self):
        pass

    @abc.abstractmethod
    def delete_data(self):
        pass
