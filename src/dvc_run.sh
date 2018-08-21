dvc add ../data/external/allyears2k.csv

dvc run -d ./models/Static_Model_Pitfalls_of_Model_Development.py -d ../data/external/allyears2k.csv \
              -o ../data/processed/baseline_model.log \
              -o ../data/processed/airlines_training_examples.pkl \
              -o ../data/processed/airlines_training_targets.pkl \
              -o ../data/processed/airlines_training_examples.pkl \
              -o ../data/processed/airlines_training_targets.pkl \
              yes | python ./models/Static_Model_Pitfalls_of_Model_Development.py | tee ../data/processed/baseline_model.log 

cat baseline_model.log.dvc
