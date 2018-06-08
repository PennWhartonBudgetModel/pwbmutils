"""PWBM Task Runner.

Adapted from SGE task runner by Jake Feala (@jfeala) from
`LSF extension <https://github.com/dattalab/luigi/blob/lsf/luigi/lsf.py>`

Expects to find a configuration file, `config.json`, in the root directory.
See ``get_config_value`` for default configuration.

The ``mem_free`` and ``n_cpu`` parameters allow you to define different
compute resource requirements (or slots, in SGE terms) for each task. In this
example, the third Task asks for 3 CPU slots. If your cluster only contains
nodes with 2 CPUs, this task will hang indefinitely in the queue. See the docs
for :class:`luigi.contrib.sge.SGEJobTask` for other SGE parameters. As for any
task, you can also set these in your luigi configuration file.
"""

# pylint: disable=E1101, W0612, C0103, W0201

import logging
import json
import os
from os.path import exists, join
import random
import shutil
from shutil import copytree
import subprocess
import sys
import time

import luigi
from luigi.contrib.hadoop import create_packages_archive
import _pickle as pickle

logger = logging.getLogger("luigi-interface")
logger.propagate = 0


def get_config_value(key):
    """Returns default configuration values for insignificant parameters.
    """

    default = {
        "cache_location": ".cache",
        "qsub_command": "qsub",
        "run_locally": 1,
        "idempotent": 0
    }

    if exists("config.json"):
        return json.load(open("config.json", "r"))[key]
    else:
        return default[key]


class PWBMTask(luigi.Task):
    """Base class for executing a job on SunGrid Engine

    Override ``work()`` (rather than ``run()``) with your job code.

    Parameters:

    - n_cpu: Number of CPUs (or "slots") to allocate for the Task. This
          value is passed as ``qsub -pe {pe} {n_cpu}``
    - parallel_env: SGE parallel environment name. The default is "orte",
          the parallel environment installed with MIT StarCluster. If you
          are using a different cluster environment, check with your
          sysadmin for the right pe to use. This value is passed as {pe}
          to the qsub command above.
    - shared_tmp_dir: Shared drive accessible from all nodes in the cluster.
          Task classes and dependencies are pickled to a temporary folder on
          this drive. The default is ``/home``, the NFS share location setup
          by StarCluster
    - job_name_format: String that can be passed in to customize the job name
        string passed to qsub; e.g. "Task123_{task_family}_{n_cpu}...".
    - job_name: Exact job name to pass to qsub.
    - run_locally: Run locally instead of on the cluster.
    - no_tarball: Don't create a tarball of the luigi project directory.  Can be
        useful to reduce I/O requirements when the luigi directory is accessible
        from cluster nodes already.
    """

    n_cpu = luigi.IntParameter(default=1, significant=False)

    shared_tmp_dir = luigi.Parameter(default="~/.tmp", significant=False)

    parallel_env = luigi.Parameter(default="openmp", significant=False)

    job_name_format = luigi.Parameter(
        significant=False,
        default="",
        description="A string that can be formatted with class variables to "
        "name the job with qsub.")

    job_name = luigi.Parameter(
        significant=False,
        default="",
        description="Explicit job name given via qsub.")

    run_locally = luigi.BoolParameter(
        default=get_config_value("run_locally"),
        significant=False,
        description="Run locally instead of on the cluster.")

    no_tarball = luigi.BoolParameter(
        significant=False,
        default=True,
        description="Don't tarball (and extract) the luigi project files")

    cache_location = luigi.Parameter(default=get_config_value("cache_location"))

    qsub_command = luigi.Parameter(default=get_config_value("qsub_command"))

    _dependencies = luigi.IntParameter(default=0)

    mem_free = "5G"

    def __init__(self, *args, **kwargs):
        super(PWBMTask, self).__init__(*args, **kwargs)
        if self.job_name != "":
            # use explicitly provided job name
            pass
        elif self.job_name_format != "":
            # define the job name with the provided format
            self.job_name = self.job_name_format.format(
                task_family=self.task_family, **self.__dict__)
        else:
            # default to the task family
            self.job_name = self.task_family


    def output(self):
        """Default output method writes to cache location.
        """

        return luigi.LocalTarget(join(self.cache_location, self.task_id))


    @classmethod
    def build_task(cls, **kwargs):
        """Factory for constructing new tasks.
        """

        task = cls(**kwargs)

        if get_config_value("idempotent") == 1:
            return task
        else:
            try:
                dependencies = hash(tuple(task.requires()))
            except TypeError:
                dependencies = hash(task.requires())
        
            return cls(**kwargs, _dependencies=dependencies)


    def run(self):
        if self.run_locally:
            return self.work()
        else:

            # Set up temp folder in shared directory (trim to max filename length)
            base_tmp_dir = self.shared_tmp_dir
            random_id = '%016x' % random.getrandbits(64)
            folder_name = self.task_id + '-' + random_id
            self.tmp_dir = os.path.join(base_tmp_dir, folder_name)
            max_filename_length = os.fstatvfs(0).f_namemax
            self.tmp_dir = self.tmp_dir[:max_filename_length]
            logger.info("Tmp dir: %s", self.tmp_dir)
            os.makedirs(self.tmp_dir)

            # Dump the code to be run into a pickle file
            logging.debug("Dumping pickled class")
            self._dump(self.tmp_dir)

            if not self.no_tarball:
                # Make sure that all the class's dependencies are tarred and available
                # This is not necessary if luigi is importable from the cluster node
                logging.debug("Tarballing dependencies")
                # Grab luigi and the module containing the code to be run
                packages = [luigi] + [__import__(self.__module__, None, None, 'dummy')]
                create_packages_archive(packages, os.path.join(self.tmp_dir, "packages.tar"))

            copytree(os.path.join("engine"), os.path.join(self.tmp_dir, "engine"))
            copytree(os.path.join("modules"), os.path.join(self.tmp_dir, "modules"))
            shutil.copyfile("config.json", os.path.join(self.tmp_dir, "config.json"))

            # make a stamp indicator in the folder
            # generate unique descriptive stamp for current commit
            get_commit_property = lambda s: subprocess.check_output(
                "git --no-pager log -1 --format=%{}".format(s), shell=True
            ).decode("utf-8").strip()

            commit_time   = get_commit_property("ci")[:16] \
                .replace(" ", "-") \
                .replace(":", "-")

            commit_author = get_commit_property("ce").split("@")[0]
            commit_hash   = get_commit_property("h")
            stamp = "{}-{}-{}".format(commit_time, commit_author, commit_hash)

            # write out stamp to temp folder
            with open(os.path.join(self.tmp_dir, "stamp"), "w") as stamp_file:
                stamp_file.write(stamp)

            # Build a qsub argument that will run sge_runner.py on the directory
            # we've specified
            runner_path = os.path.join("utility", "sge_runner.py")
            if runner_path.endswith("pyc"):
                runner_path = runner_path[:-3] + "py"
            # enclose tmp_dir in quotes to protect from special escape chars
            job_str = 'python {0} "{1}" "{2}"'.format(
                runner_path, self.tmp_dir, os.getcwd())
            if self.no_tarball:
                job_str += ' "--no-tarball"'

            # Build qsub submit command
            self.outfile = os.path.join(self.tmp_dir, 'job.out')
            self.errfile = os.path.join(self.tmp_dir, 'job.err')

            qsub_template = """echo {cmd} | {qsub_command} -o ":{outfile}" -e ":{errfile}" -V -r y -pe {pe} {n_cpu} -N {job_name} -l m_mem_free={mem_free} -sync y"""

            submit_cmd = qsub_template.format(
                cmd=job_str,
                job_name=self.job_name,
                outfile=self.outfile,
                errfile=self.errfile,
                pe=self.parallel_env,
                n_cpu=self.n_cpu,
                mem_free=self.mem_free,
                qsub_command=self.qsub_command
            )

            logger.debug('qsub command: \n' + submit_cmd)

            # Submit the job and grab job ID
            try:
                output = subprocess.check_output(
                    submit_cmd,
                    shell=True,
                    stderr=subprocess.STDOUT
                )

                logger.debug("qsub job complete with output:\n" + output.decode('utf-8'))
            except subprocess.CalledProcessError as e:
                logger.error("qsub submission failed with output:\n" + e.output.decode('utf-8'))
                if os.path.exists(os.path.join(self.tmp_dir, "job.err")):
                    with open(os.path.join(self.tmp_dir, "job.err"), "r") as err:
                        logger.error(err.read())

            # delete the temporaries, if they're there.
            if self.tmp_dir and os.path.exists(self.tmp_dir):
                logger.info('Removing temporary directory %s', self.tmp_dir)
                subprocess.call(["rm", "-rf", self.tmp_dir])

            # wait a beat, to give things a chance to settle
            time.sleep(5)

            # check whether the file exists
            if not os.path.exists(self.output().path):
                raise Exception("qsub failed to produce output")

    def work(self):
        """Override this method, rather than ``run()``,  for your actual work."""
        pass

    def _dump(self, out_dir=''):
        """Dump instance to file."""
        with self.no_unpicklable_properties():

            self.job_file = os.path.join(out_dir, 'job-instance.pickle')

            logger.info("Dumping module...")

            if self.__module__ == '__main__':
                d = pickle.dumps(self)
                module_name = os.path.basename(sys.argv[0]).rsplit('.', 1)[0]
                d = d.replace(b'c__main__', b"c" + module_name.encode('utf-8'))
                open(self.job_file, "wb").write(d)
            else:
                pickle.dump(self, open(self.job_file, "wb"))

    @classmethod
    def get_param_defaults(cls):
        all_params = cls.get_params()
        return all_params
