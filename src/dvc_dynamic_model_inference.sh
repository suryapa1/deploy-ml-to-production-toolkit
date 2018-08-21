dvc add ../data/external/allyears2k.csv

dvc run -d ./models/Dynamic_Model_Inference.py -d ../data/external/allyears2k.csv \
              -o ../data/processed/dynamic_model_inference.log \
              -o ../data/processed/airlines_training_examples.pkl \
              -o ../data/processed/airlines_training_targets.pkl \
              -o ../data/processed/airlines_training_examples.pkl \
              -o ../data/processed/airlines_training_targets.pkl \
              yes | python ./models/Dynamic_Model_Inference.py | tee ../data/processed/dynamic_model_inference.log 

cat ../data/processed/dynamic_model_inference.log
