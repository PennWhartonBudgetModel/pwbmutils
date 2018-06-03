"""Collection of basic tests of utilities.
"""

import os
from unittest import TestCase

import luigi
import pandas

import pwbmutils

class TestUtils(TestCase):
    """Collection of basic tests of utilities.
    """

    def test_interface_reader(self):
        """Tests whether the interface reader reads from the HPCC.
        """

        # read the projections interface
        test_task = pwbmutils.InterfaceReader(
            interface_info={
                "version": "2018-05-21-21-55-arnon-d43fbc6",
                "interface": "projections",
                "component": "Projections"
            },
            filename="Projections.csv"
        )

        test_task.run()

        projections = pandas.read_csv(test_task.output().path)

        self.assertTrue(len(projections) > 0)


    def test_example_task(self):
        """Test whether the example task produces output.
        """

        test_task = pwbmutils.ExampleTask.build_task()

        luigi.build([test_task], local_scheduler=True)

        projections = pandas.read_csv(
            os.path.join(
                test_task.output().path,
                "projections.csv"
            )
        )

        self.assertTrue(len(projections) > 0)


    def test_interface_writer(self):
        """Test whether the interface writer writes out successfully.
        """

        example_task = pwbmutils.ExampleTask.build_task()
        test_task = pwbmutils.InterfaceWriter.build_task(
            output_task=example_task,
            name_of_component="TestUtilities",
            name_of_interface="testinterface")

        luigi.build([test_task], local_scheduler=True)

        projections = pandas.read_csv(
            os.path.join(
                test_task.output().path,
                "projections.csv"
            )
        )

        self.assertTrue(len(projections) > 0)
    