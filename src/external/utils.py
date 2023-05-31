import os
import sys
import git
import subprocess

import logging


def import_git_module(name, git_url, patch=None):
    assert name in git_url
    tmp_dir = os.path.join("/tmp", "recheck", name)
    try:
        r = git.Repo.clone_from(
            git_url,
            tmp_dir,
        )
        logging.info(f"Successfully, cloned the GitHub module: {git_url}")
    except:
        logging.error(
            "Could not clone the repo. This might lead to errors in later stages."
        )

    if patch:
        try:
            r.git.execute(
                [
                    "git",
                    "apply",
                    patch,
                ]
            )
            logging.info("Successfully applied the patch.")
        except:
            logging.error(
                "Could not apply the patch. This might lead to errors in the later stages."
            )

    try:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-r",
                os.path.join(tmp_dir, "requirements.txt"),
            ],
            stdout=subprocess.DEVNULL,
        )
        logging.info("Successfully installed requirements.txt")
    except:
        logging.error(
            f"Could not install requirements.txt -> might be missing from: {tmp_dir}"
        )

    sys.path.insert(0, tmp_dir)
    return
