import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import argparse
from runTests import run_tests
from PIL import Image
import numpy as np


def runHw2():
    # runHw2 is the "main" interface that lets you execute all the 
    # walkthroughs and challenges in this homework. It lists a set of 
    # functions corresponding to the problems that need to be solved.
    #
    # Note that this file also serves as the specifications for the functions 
    # you are asked to implement. In some cases, your submissions will be 
    # auto-graded.  Thus, it is critical that you adhere to all the specified 
    # function signatures.
    #
    # Before your submission, make sure you can run runHw2('all') 
    # without any error.
    #
    # Usage:
    # python runHw2.py                  : list all the registered functions
    # python runHw2.py 'function_name'  : execute a specific test
    # python runHw2.py 'all'            : execute all the registered functions
    parser = argparse.ArgumentParser(
        description='Execute a specific test or all tests.')
    parser.add_argument(
        'function_name', type=str, nargs='?', default='all',
        help='Name of the function to test or "all" to execute all the registered functions')
    args = parser.parse_args()

    # Call test harness
    fun_handles = {'honesty': honesty, 
                   'walkthrough1': walkthrough1,
                   'challenge1a': challenge1a, 
                   'challenge1b': challenge1b, 
                   'challenge1c': challenge1c,
                   'demoTricks': demoTricks}
    run_tests(args.function_name, fun_handles)


# Academic Honesty Policy
def honesty():
    from signAcademicHonestyPolicy import sign_academic_honesty_policy
    # Type your full name and uni (both in string) to state your agreement 
    # to the Code of Academic Integrity.
    sign_academic_honesty_policy('Peter Parker', 'pp117')


# Test for Walkthrough 1: Morphological operations
def walkthrough1():
    from hw2_walkthrough1 import hw2_walkthrough1
    hw2_walkthrough1()



from hw2_challenge1 import hw2_challenge1a,hw2_challenge1b,hw2_challenge1c
# Tests for Challenge 1: 2D binary object recognition
def challenge1a():
    hw2_challenge1a()


def challenge1b():
    hw2_challenge1b()


def challenge1c():
    hw2_challenge1c()


def demoTricks():
    from demoTricksFun import demo
    demo()


if __name__ == '__main__':
    runHw2()