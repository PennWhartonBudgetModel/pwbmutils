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
import portalocker

logger = logging.getLogger('luigi-interface')

class MapTarget(Target):
    """Target which also maps itself for external consumption using a map file.

    Extends the notion of a file system target by also incorporating a parameter
    value, which is stored in a map file, and used to retrieve the indexed file.
    """

    def __init__(
            self,
            base_path,
            params,
            hash_value,
            map_name="map.csv",
            max_timeout=2400
        ):
        """Initializes a new map target.
        
        Arguments:
            base_path {str} -- The path to store the map file and subfolders.
            params {dict} -- Dictionary of parameters to map. Must correspond to
            the values in the map file already created, otherwise, an exception
            will be raised.
        
        Keyword Arguments:
            map_name {str} -- Name of the map file. (default: {"map.csv"})
            id_name {str} -- Name of the id column in the map file. (default: {"id"})
            max_timeout {int} -- Maximum number of seconds to wait for a lock to resolve. (default: {2400})
        """

        self.base_path = base_path
        self.params = params
        self.map_name = map_name
        self.tmp_dir = None
        self.map = None
        self.hash = str(hash_value)
        self.max_timeout = max_timeout


    def __enter__(self):

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

        return self

    def __exit__(self, type, value, traceback):

        try:

            with portalocker.Lock(os.path.join(self.base_path, self.map_name), 'a') as map_handle:

                # construct a new id
                new_id = self.hash

                # create a new table to append, with the new id, and the parameters
                new_entry = pandas.DataFrame({
                    k: [self.params[k]] for k in self.params
                })
                new_entry[""] = new_id
                new_entry[""] = new_entry[""].astype(str)
                new_entry = new_entry.set_index("")

                # remove the directory, if it exists
                if os.path.exists(os.path.join(self.base_path, str(new_id))):
                    shutil.rmtree(os.path.join(self.base_path, str(new_id)))

                # move the temporary directory
                os.rename(
                    self.tmp_dir,
                    os.path.join(self.base_path, str(new_id))
                )

                # write the new map file out
                if os.stat(os.path.join(self.base_path, self.map_name)).st_size == 0:
                    new_entry.to_csv(
                        map_handle
                    )
                else:
                    new_entry.to_csv(
                        map_handle,
                        header=False
                    )
        except:
            pass
        finally:
            if os.path.exists(self.tmp_dir):
                shutil.rmtree(self.tmp_dir)


    def exists(self):
        """Returns True if file exists and parameters exist in map file.
        """

        # if no map file exists, then this target does not exist
        if not os.path.exists(os.path.join(self.base_path, self.map_name)):
            return False

        # read the map file
        self.map = pandas.read_csv(os.path.join(self.base_path, self.map_name), index_col=0)

        # if it's in the map file and the index, then it exists
        if self.hash in self.map.index and os.path.exists(os.path.join(self.base_path, self.hash)):
            return True
        
        # if it's in neither, then it does not exist
        if self.hash not in self.map.index and not os.path.exists(os.path.join(self.base_path, self.hash)):
            return False
        
        # otherwise, something went wrong
        if self.hash in self.map.index and not os.path.exists(os.path.join(self.base_path, self.hash)):
            raise luigi.parameter.ParameterException(
                "Index %s in map file but not in folder" % self.hash
            )

        if self.hash not in self.map.index and os.path.exists(os.path.join(self.base_path, self.hash)):
            raise luigi.parameter.ParameterException(
                "Index %s in folder but not in map file" % self.hash
            )
