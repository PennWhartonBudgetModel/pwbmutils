"""Collection of basic tests of utilities.
"""

import os
import shutil
from unittest import TestCase

import luigi
import numpy
import pandas

import pwbmutils
import pwbmutils.statistics

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


    def test_interface_direct_writer(self):
        """Test whether the interface direct writer writes out successfully.
        """

        example_task = pwbmutils.ExampleTask.build_task()
        test_task = pwbmutils.InterfaceDirectWriter.build_task(
            output_folder=example_task.output().path,
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


    def test_map_target(self):

        if os.path.exists("test_interface"):
            shutil.rmtree("test_interface")

        os.makedirs("test_interface")
    
        map_target = pwbmutils.MapTarget(
            "test_interface",
            {
                "Param1": 2,
                "Param2": 10
            }
        )

        # map target should not exist
        self.assertFalse(map_target.exists())

        # output to write
        output = pandas.DataFrame({"test": [1, 2, 3, 4]})

        # write to map target
        with map_target as out:
            output.to_csv(os.path.join(out.tmp_dir, "test.csv"))

        # check that target exists
        self.assertTrue(map_target.exists())

        # make another map target
        map_target = pwbmutils.MapTarget(
            "test_interface",
            {
                "Param1": 6,
                "Param2": 1
            }
        )

        # map target should not exist
        self.assertFalse(map_target.exists())

        # write to map target
        with map_target as out:
            output.to_csv(os.path.join(out.tmp_dir, "test.csv"))

        # check that target exists
        self.assertTrue(map_target.exists())


    def test_statistics(self):
        """Test whether the interface writer writes out successfully.
        """

        dataset = pandas.DataFrame({
            "Year": [1996, 1997, 1998, 1999, 2000],
            "Value": [0, 1, 1, 0, 0]
        })

        linear_model = pwbmutils.statistics.LinearRegression(
            "Value~Year",
            data=dataset
        )

        self.assertTrue(
            numpy.sum(
                numpy.abs(
                    linear_model.predict(dataset) \
                    - numpy.array([0.6, 0.5, 0.4, 0.3, 0.2])
                )
            ) < 10e-6
        )

        logit_model = pwbmutils.statistics.LogitRegression(
            "Value~Year",
            data=dataset
        )

        self.assertTrue(
            numpy.sum(
                numpy.abs(
                    logit_model.predict(dataset) \
                    - numpy.array([
                        0.60717446,
                        0.49898785,
                        0.39089593,
                        0.29254677,
                        0.210395
                    ])
                )
            ) < 10e-6
        )