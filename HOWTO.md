# HOWTO

## using run_test.sh

This file must be run in a bash environment, VSCode has a virtual bash terminal which is what I use
It is used to check if the answer sources in leet.py pass the cases tested in test.py

## using get_sol.sh

This file also must be run in a bash environment
This file is used to retrieve the solution for a specified question by the number or function name

**use case example:**

`$ bash ./get_sol.sh 12`

or

`$ bash ./get_sol.sh isPowerOfFour`

## using test.py

If run it will run some basic test cases and will print if there are any errors or not, with a list

of failures and errors in two seprate lines

## using solved.csv

This file can be used to manually search for solved questions, and it's info, like

question number, difficulty, function name, question name, and topics

## using leet.py

This file is only for the source of the answers
