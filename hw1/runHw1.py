import argparse
from runTests import run_tests
from signAcademicHonestyPolicy import sign_academic_honesty_policy
from hw1_walkthrough1 import hw1_walkthrough1
from hw1_walkthrough2 import hw1_walkthrough2
from hw1_walkthrough3 import hw1_walkthrough3


def runHw1():
    # runHw1 is the "main" interface that lets you execute all the
    # walkthroughs and challenges in this homework. It lists a set of
    # functions corresponding to the problems that need to be solved.
    #
    # Note that this file also serves as the specifications for the functions
    # you are asked to implement. In some cases, your submissions will be
    # auto-graded. Thus, it is critical that you adhere to all the specified
    # function signatures.
    #
    # Before your submssion, make sure you can run runHw1('all')
    # without any error.
    #
    # Usage:
    # python runHw1.py                  : list all the registered functions
    # python runHw1.py 'function_name'  : execute a specific test
    # python runHw1.py 'all'            : execute all the registered functions
    parser = argparse.ArgumentParser(
        description='Execute a specific test or all tests.')
    parser.add_argument(
        'function_name', type=str, nargs='?', default='all',
        help='Name of the function to test or "all" to execute all the registered functions')
    args = parser.parse_args()

    # Call test harness
    fun_handles = {'honesty': honesty,
                   'walkthrough1': walkthrough1,
                   'walkthrough2': walkthrough2,
                   'walkthrough3': walkthrough3}
    run_tests(args.function_name, fun_handles)


def honesty():
    # Type your full name and uni (both in string) to state your agreement
    # to the Code of Academic Integrity.
    sign_academic_honesty_policy('Huzaifa Mustafa Unjhawala', 'unjhawala')


def walkthrough1():
    # Open hw1_walkthrough1.py and go through a short Python tutorial
    # Feel free to try the commands or variations of them at your
    # MATLAB command prompt if you are not familiar with MATLAB yet. You are
    # not required to submit any code for this Walkthrough 1.
    hw1_walkthrough1()


def walkthrough2():
    # Fill in the partially complete code in hw1_walkthrough2.py
    # Submit the completed code and the outputs
    hw1_walkthrough2()


def walkthrough3():
    # Fill in the partially complete code in hw1_walkthrough3.m.
    # Submit the completed code and the outputs
    hw1_walkthrough3()


if __name__ == "__main__":
    runHw1()
