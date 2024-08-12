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
    if [ $(wc -l < "$filename") -ge $1 ]; then
        echo $( head -n $1 "$filename" | tail -n 1 )
    fi
}

for Line in $fails_error_array ; do
    echo $( extractLine $(( $Line - 1)) testfile.txt )
    echo $( extractLine $Line testfile.txt )
    echo ""

done
