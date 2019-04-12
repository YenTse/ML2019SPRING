wget -O ./model.zip https://www.dropbox.com/sh/a6wt9yrntwdt6km/AAA4pdz7tr3_8v7FzuJXzcCza?dl=1
unzip ./model.zip
rm ./model.zip
python3 hw3_v21_ensemble_predict.py $1 $2

