

# /home/local/eliel/venvs/ve37/bin/python3 eunlg/bulk_generate.py -l en -o bulk_out/ -v full -d cphi --locations DE DK EE EL ES FI
# /home/local/eliel/venvs/ve37/bin/python3 eunlg/bulk_generate.py -l en -o bulk_out/ -v full -d health_funding --locations DE DK EE EL ES FI
# /home/local/eliel/venvs/ve37/bin/python3 eunlg/bulk_generate.py -l en -o bulk_out/ -v full -d health_cost --locations DE DK EE EL ES FI
# /home/local/eliel/venvs/ve37/bin/python3 eunlg/bulk_generate.py -l en -o bulk_out/ -v score -d cphi --locations DE DK EE EL ES FI
# /home/local/eliel/venvs/ve37/bin/python3 eunlg/bulk_generate.py -l en -o bulk_out/ -v score -d health_funding --locations DE DK EE EL ES FI

/home/local/eliel/venvs/ve37/bin/python3 eunlg/bulk_generate.py -l en -o pos+LSTM_l50_0.1-10 -v score -d health_cost --locations FI --verbose
/home/local/eliel/venvs/ve37/bin/python3 eunlg/bulk_generate.py -l en -o pos+LSTM_l50_0.1-10 -v baseline -d health_cost --locations FI --verbose
/home/local/eliel/venvs/ve37/bin/python3 eunlg/bulk_generate.py -l en -o pos+LSTM_l50_0.1-10 -v random -d health_cost --locations FI --verbose
/home/local/eliel/venvs/ve37/bin/python3 eunlg/bulk_generate.py -l en -o pos+LSTM_l50_0.1-10 -v full -d health_cost --locations FI --verbose
