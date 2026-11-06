# Baselines (tables behind Table 3, Table 5, Fig. 7, Fig. 8 and Tables S7–S8)

Classical and process-based baselines on exactly the inputs, targets and splits of the notebooks.
They run on a CPU (about 15 min in total) and write to `results/baselines/`.

```bash
cd baselines
python prep_features.py chrono      # features of notebook 01 (no training)
python prep_features.py season      # features of notebook 02
python run_ml.py                    # persistence, ARX, random forest, gradient boosting
python run_phys.py chrono free      # Richards solver without IoT update
python run_phys.py chrono onestep   # Richards solver updated with every IoT reading
python run_phys.py season free
python run_phys.py season onestep
python run_transfer_ml.py           # ARX and random forest, transfer to Dev107
```

`richards1d.py` is the 1-D Richards solver (mixed form, modified Picard; same domain, van Genuchten–Mualem
parameters, rain flux capped at Ks and free-drainage base as the network). `prep.py` contains the data-preparation
functions of the notebook.
