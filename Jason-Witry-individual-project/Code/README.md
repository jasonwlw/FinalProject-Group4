Dr. Jafari, to recreate the weird spike in validation loss for ResNet_fc6;
Run:

python witry_fc8.py > ResNet_fc8_20E.log

python retrain_fc8.py > ResNet_fc8_60E.log

cat ResNet_fc8_20E.log ResNet_fc8_60E.log > ResNet_fc8.log

python loss_plots.py

and the results should be there
