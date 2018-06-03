"""Makes available the code for PWBM components.
"""

# pylint: disable=E1136

import logging
import os
import sys
from os import makedirs
from os.path import join, exists
import shutil

import git
import luigi
from .pwmb_task import PWBMTask

class ComponentCode(PWBMTask):
    """Checks out requested code component from github and puts it in
    task directory.
    """

    interface_info = luigi.DictParameter()

    def output(self):
        return luigi.LocalTarget(join(self.cache_location, self.task_id))

    def requires(self):
        return []
    
    def work(self):
        temp_task_id = self.task_id + "_tmp"

        if exists(join(self.cache_location, temp_task_id)):
            shutil.rmtree(join(self.cache_location, temp_task_id))
        
        if not exists(join(self.cache_location, temp_task_id)):
            makedirs(join(self.cache_location, temp_task_id))

        # download a compressed archive to temporary location
        repo_hash = self.interface_info["version"][-7:]

        author_abb = self.interface_info["version"].split("-")[-2]

        if author_abb == "njanetos":
            author = "nick"
        elif author_abb == "efraim":
            author = "efraim"
        elif author_abb == "aherrick":
            author = "austin"

        # clone repo
        git.Git(
            join(self.cache_location, temp_task_id)
        ).clone(
            "https://{u}:{p}@{site}/{org}/{component}-{author}.git".format(
                u="njanetos-pwbm",
                p="0b022b922b061d7279e6c60ac8bc08cdd4742dbb",
                site="github.com",
                org="PennWhartonBudgetModel",
                component=self.interface_info["component"],
                author=author
            )
        )

        # reset to requested hash
        temp_repo_dir = join(self.cache_location, temp_task_id)
        git.Repo(join(temp_repo_dir, self.interface_info["component"]+"-"+author)).git.reset("--hard", repo_hash)

        # copy everything in the repo one level up so it's not under a
        # confusing hash
        repo_file = os.listdir(temp_repo_dir)[0]
        os.rename(join(temp_repo_dir, repo_file), join(temp_repo_dir, "repo"))

        # move folder to the final output location
        self.copytree(temp_repo_dir, self.output().path)

        # delete temporary folder
        try:
            shutil.rmtree(temp_repo_dir)
        except Exception as e:
            logging.warning("Failed to remove temporary repo file")

    def copytree(self, src, dst, symlinks=False, ignore=None):
        """Taken from
        https://stackoverflow.com/questions/1868714/how-do-i-copy-an-entire-directory-of-files-into-an-existing-directory-using-pyth
        """

        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.exists(d):
                try:
                    shutil.rmtree(d)
                except Exception as e:
                    os.unlink(d)
            if os.path.isdir(s):
                shutil.copytree(s, d, symlinks, ignore)
            else:
                shutil.copy2(s, d)
