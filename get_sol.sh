#!bin/bash

# check if the passed argument is the question number or function name
if [[ $1 =~ [0-9]+$ ]]; then
    Field=1
    prefix="  #"
else
    Field=2
    prefix="def"
fi

# get the line number of $1 function using grep -n $1 leet.py  **
fgrep -w -n "$prefix $1" leet.py > /dev/null

if [[ $? -gt 0 ]]; then
    echo "question not found"
    exit 1
else
    echo `fgrep -w -n "$prefix $1" leet.py` > temp.txt
    line_i=`cut -d ":" -f 1 temp.txt`
    rm temp.txt
fi
# echo $line_i
# get the next number of line for the next function
# reading solved.csv
line_f=`grep -w -n $1 solved.csv | cut -d ":" -f 1`
line_f=$(($line_f + 1))

next_func=`head -n $line_f solved.csv | tail -n 1 | cut -d "," -f $Field` # if passes the function name: -f 2
if [[ $next_func -eq $1 ]] ; then
    line_f=`wc -l leet.py | cut -d " " -f 6`
else
    line_f=`grep -w -n $next_func leet.py | cut -d ":" -f 1`
fi
# print the lines in between
head -n $(($line_f - 1)) leet.py | tail -n $(($line_f - $line_i))

echo "Do you want the output in sol_"$1".py? (y/n)"
read answer
if [ $answer = "y" ]; then
    clear
    filename="sol_"$1".py"
    touch $filename
    echo "from leet import *

class Solution:" > $filename
    head -n $(($line_f - 1)) leet.py | tail -n $(($line_f - $line_i)) >> $filename
    echo "Done!"
fi
