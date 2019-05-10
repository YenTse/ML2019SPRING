wget -O ./hw6.zip https://www.dropbox.com/sh/mpp3nhbwceo0xx8/AADUnUtH6iFOfmDkPX_n2NZIa?dl=1

unzip -o ./hw6.zip

rm ./hw6.zip

python3 hw6_test_RNN.py $1 $2 $3

