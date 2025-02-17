#!bin/bash
clear
cat test.py > testfile.txt

# want to use this to find the failed test, test case and the AssertionError

result=`python3 test.py`
clear

LEN=${#result}  # gets the length of a string do ${#reuslt[@]} if a array
START=0
fails_error_array=()

if [[ $result != *"line"* ]]; then  # to use wildcards do [[]]
    echo "Done successfuly!"
    rm testfile.txt
    exit
fi


# to extract the line numbers from result
while [ $START -lt $LEN ]; do  # the logical operators for integers and strings are different
    if [ "${result:$START:5}" = "line " ]; then

        START=$(($START + 5))
        END=$(($START+1))

        while [ ${result[$END]} != "," ]; do
            END=$(($END + 1))  # simple expresion is done in $(())
        done

        fails_error_array+=( $(( ${result:$START:$(($END - $START + 2))} + 0)) )  # how to add to arrays
    fi

    START=$(($START + 1))
done
clear

function extractLine {  # number of the line, filename
    filename=$2
    if [ $(wc -l < "$filename") -ge $1 ]; then  # this checks if the number of lines in filename is ge that $1
        echo $( head -n $1 "$filename" | tail -n 1 )  # first take the first $1 number of line in filenames and
        # uses | to pass it to tail command which outputs the last line
    fi
}

for Line in $fails_error_array ; do
    echo $( extractLine $(( $Line - 1)) testfile.txt )
    echo $( extractLine $Line testfile.txt )
    echo ""

done

rm testfile.txt

echo "Would you like to see the detailed failed test case(s)? (y/n) "
read answer
if [ $answer = "y" ]; then
    python3 -m unittest test.Test
else
    clear
fi


# $? is the exit code of the most recent command
# $PATH is the paths terminal looks for commands
# export
