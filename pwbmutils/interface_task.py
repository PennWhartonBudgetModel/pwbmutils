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
from .map_target import MapTarget

# pylint: disable=E1101

class InterfaceTask(PWBMTask):

	stamp = luigi.BoolParameter(
		default=subprocess.call(["git", "diff-index", "--quiet", "HEAD", "--"]) not in [1, 128] or os.path.exists("stamp"),
		description="Whether to perform a stamped run. If true, will check for "
		"uncommitted changes, and if none exist, will create a descriptive "
		"stamp and write to the HPCC server.")

	path_to_hpcc = luigi.Parameter(
		default=r"\\hpcc-ppi.wharton.upenn.edu\ppi" \
			if sys.platform == "win32" \
			else "/home/mnt/projects/ppi",
		description="Path to the HPCC.")

	name_of_component = luigi.Parameter(
		description="The component name.")

	name_of_interface = luigi.Parameter(
		description="The name of the interface to write to.")

	public_parameters = luigi.DictParameter(
		description="Public-facing parameters to be written to map file.")

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

			return MapTarget(
				join(
					self.path_to_hpcc,
					self.name_of_component,
					"Interfaces",
					stamp,
					self.name_of_interface
				),
				self.public_parameters,
				self.task_id
			)

		else:

			return MapTarget(
				join(
					self.cache_location,
					"Interfaces",
					self.name_of_interface
				),
				self.public_parameters,
				self.task_id
			)
