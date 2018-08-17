"""Target which also maps itself for external consumption using a map file.
"""

import datetime
import logging
import os
import shutil
import time

import luigi
from luigi.target import Target
import pandas

logger = logging.getLogger('luigi-interface')


class MapTarget(Target):

    def __init__(self, base_path, params, map_name="map.csv", id_name="id"):
        """
        Initializes a FileSystemTarget instance.

        :param str path: the path associated with this FileSystemTarget.
        :param dict params: the parameter set to write out to a map file.
        """
        self.base_path = base_path
        self.params = params
        self.map_name = map_name
        self.id_name = id_name
        self.tmp_dir = None
        self.map = None


    def __enter__(self):
        # check for a lock file, if it exists, wait until it's gone to continue
        while True:
            if os.path.exists(os.path.join(self.base_path, "lock")):
                time.sleep(1)
            else:
                os.makedirs(os.path.join(self.base_path, "lock"))
                break

        # define a temporary directory using current date
        self.tmp_dir = os.path.join(
            self.base_path,
            "tmp-dir-%{date}".format(
                date=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
            )
        )

        # delete the temporary directory, if it exists
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

        # create the temporary directory
        os.makedirs(self.tmp_dir)

        # load the map file, create if it does not exist
        if not os.path.exists(os.path.join(self.base_path, self.map_name)):
            self.map = pandas.DataFrame(columns=list(self.params.keys()) + [self.id_name])
        else:
            self.map = pandas.read_csv(os.path.join(self.base_path, self.map_name))

        return self

    def __exit__(self, type, value, traceback):

        # construct a new id, 0 if no existing entries
        if len(self.map) == 0:
            new_id = 0
        else:
            new_id = max(self.map[self.id_name].values) + 1

        # create a new table to append, with the new id, and the parameters
        new_entry = pandas.DataFrame({
            k: [self.params[k]] for k in self.params
        })
        new_entry[self.id_name] = new_id
        self.map = self.map.append(new_entry)
        
        # remove the directory
        if os.path.exists(os.path.join(self.base_path, str(new_id))):
            shutil.rmtree(os.path.join(self.base_path, str(new_id)))

        # move the temporary directory
        os.rename(
            self.tmp_dir,
            os.path.join(self.base_path, str(new_id))
        )

        # write the new map file out
        self.map.to_csv(
            os.path.join(self.base_path, self.map_name),
            index=False
        )

        # lift the lock
        shutil.rmtree(os.path.join(self.base_path, "lock"))


    def exists(self):
        """Returns True if file exists and parameters exist in map file.
        """

        # if no map file exists, then this target does not exist
        if not os.path.exists(os.path.join(self.base_path, self.map_name)):
            return False

        # read the map file
        self.map = pandas.read_csv(os.path.join(self.base_path, self.map_name))

        # check that an id name column exists in the map file
        if not self.id_name in self.map:
            raise luigi.parameter.ParameterException("Id column named %s was "
                                                     "not found in map file "
                                                     "%s" % (self.id_name, self.map_name))

        # select the right row from the map file
        _map = self.map.copy()
        for key in self.params:
            if not key in _map:
                raise luigi.parameter.ParameterException("Missing key %s in "
                                                         "map file." % key)

            _map = _map[_map[key] == self.params[key]].reset_index(drop=True)

        # check that it uniquely identifies an id
        if len(_map) > 1:
            raise luigi.parameter.ParameterException("Parameter set %s does "
                                                     "not uniquely identify a "
                                                     "parameter." % self.params)

        # if no entry exists, then target does not exist
        if len(_map) == 0:
            return False

        # if an entry exists in the map file, check that the associated file
        # also exists
        _id = _map[self.id_name].values[0]

        return os.path.exists(os.path.join(self.base_path, str(_id)))
