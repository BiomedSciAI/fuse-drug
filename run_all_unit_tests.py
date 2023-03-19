"""
This script searches recursively for unit tests and generates the tests results in xml format in the way that Jenkins expects.
In the case that it's a Jenkins job, it should delete any created cache (not implemented yet)
"""

import logging
import sys
import unittest
import os
import termcolor
import xmlrunner

print(os.path.dirname(os.path.realpath(__file__)))


def mehikon(a, b):  # type: ignore
    print(a)


termcolor.cprint = mehikon  # since junit/jenkins doesn't like text color ...

if __name__ == "__main__":
    mode = None
    if len(sys.argv) > 1:
        mode = sys.argv[1]  # options "examples", "core" or None for both "core" and "examples"
    os.environ["DISPLAY"] = ""  # disable display in unit tests

    is_jenkins_job = "WORKSPACE" in os.environ and len(os.environ["WORKSPACE"]) > 2

    search_base = os.path.dirname(os.path.realpath(__file__))
    output = f"{search_base}/test-reports/"
    print("will generate unit tests output xml at :", output)

    sub_sections_core = [("fusedrug", search_base)]
    sub_sections_examples = [("fusedrug_examples/tests", search_base)]

    if mode is None:
        sub_sections = sub_sections_core + sub_sections_examples
    elif mode == "core":
        sub_sections = sub_sections_core
    elif mode == "examples":
        sub_sections = sub_sections_examples
    else:
        raise Exception(f"Error: unexpected mode {mode}")

    suite = None
    for curr_subsection, top_dir in sub_sections:
        curr_subsuite = unittest.TestLoader().discover(
            f"{search_base}/{curr_subsection}", "test*.py", top_level_dir=top_dir
        )
        if suite is None:
            suite = curr_subsuite
        else:
            suite.addTest(curr_subsuite)

    # enable fuse-drug logger and avoid colors format
    lgr = logging.getLogger("FuseDrug")
    lgr.setLevel(logging.INFO)

    test_results = xmlrunner.XMLTestRunner(output=output, verbosity=2, stream=sys.stdout).run(
        suite,
    )

    from os import listdir
    from os.path import isfile, join

    examples_test_files = [
        f for f in listdir(output) if (isfile(join(output, f)) and f.startswith("TEST-fusedrug_examples"))
    ]

    for file in examples_test_files:
        file_path = os.path.join(output, file)

        # Open the file for reading
        with open(file_path, "r") as f:
            # Read the contents of the file
            contents = f.read()

        # Replace all occurrences of "a" with "b"
        contents = contents.replace("\u001b", "?")

        # Open the file for writing (this will overwrite the original file)
        with open(file_path, "w") as f:
            # Write the modified contents back to the file
            f.write(contents)
