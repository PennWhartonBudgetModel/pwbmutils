"""Contains luigi class writing to the HPCC and creating 'stamped' runs, of the
output of a luigi task.
"""

__author__ = "Nick Janetos"
__copyright__ = "2018 Penn Wharton Budget Model"

import os
from os.path import exists, join
import shutil
import sys
import subprocess

import luigi

from .pwbm_task import PWBMTask

# pylint: disable=E1101

class InterfaceWriter(PWBMTask):
    """Task for creating stamped runs that are moved to the HPCC, available to
    other components.

    Parameters:
        BoolParameter {stamp} -- True, to perform a stamped run.
        TaskParameter {output_task} -- Luigi task to copy up to HPCC.
        Parameter {name_of_component} -- The name of the component to copy to.
        Parameter {name_of_interface} -- The name of the interface to copy to.

    Raises:
        Exception -- Thrown if uncommitted changes in repository.
    """

    stamp = luigi.BoolParameter(
        default=False,
        description="Whether to perform a stamped run. If true, will check for "
        "uncommitted changes, and if none exist, will create a descriptive "
        "stamp and write to the HPCC server.")

    output_task = luigi.TaskParameter(
        description="Task to output to HPCC.")

    path_to_hpcc = luigi.Parameter(
        default=r"\\hpcc-ppi.wharton.upenn.edu\ppi" \
            if sys.platform == "win32" \
            else "/home/mnt/projects/ppi",
        description="Path to the HPCC.")

    name_of_component = luigi.Parameter(
        description="The component name.")

    name_of_interface = luigi.Parameter(
        description="The name of the interface to write to.")

    def output(self):
        """If performing a stamped run, will write to HPCC. Otherwise, will
        write to local cache location under Interfaces folder.

        Raises:
            Exception -- Thrown if uncommitted changes in the repository.
        """

        if self.stamp:

            # check for uncommitted changes and throw an exception if found
            result = subprocess.call([
                "git",
                "diff-index",
                "--quiet",
                "HEAD", "--"
            ])

            if result == 1:
                raise Exception(
                    "Uncommitted changes in repository, commit before stamped "
                    "run."
                )

            # generate unique descriptive stamp for current commit
            # if running on the HPCC, this needs to be loaded from a stamp
            # file, copied to local folder, since a copy is made for each task
            # of the code and a file called 'stamp' placed in it
            if self.run_locally or not os.path.exists("stamp"):
                get_commit_property = lambda s: subprocess.check_output(
                    "git --no-pager log -1 --format=%{}".format(s),
                    shell=True
                ).decode("utf-8").strip()

                commit_time = get_commit_property("ci")[:16] \
                    .replace(" ", "-") \
                    .replace(":", "-")

                commit_author = get_commit_property("ce").split("@")[0]
                commit_hash = get_commit_property("h")
                stamp = "{}-{}-{}".format(
                    commit_time,
                    commit_author,
                    commit_hash
                )

            else:
                with open("stamp", 'r') as file_in:
                    stamp = file_in.read()

            return luigi.LocalTarget(join(
                self.path_to_hpcc,
                self.name_of_component,
                "Interfaces",
                stamp,
                self.name_of_interface,
                self.output_task.task_id
            ))

        else:

            return luigi.LocalTarget(
                join(
                    self.cache_location,
                    "Interfaces",
                    self.task_id,
                    self.name_of_interface
                )
            )


    def requires(self):
        return self.output_task


    def work(self):

        if not exists(self.output().path):
            os.makedirs(self.output().path)

        os.rmdir(self.output().path)

        # move output directory to final directory
        shutil.move(
            self.output_task.output().path,
            self.output().path
        )
