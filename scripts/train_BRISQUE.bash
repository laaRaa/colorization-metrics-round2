#! /bin/bash
python src/colorization_metrics/metrics/train_brisque.py -m original -c 1024 -e 2.946847108941092
python src/colorization_metrics/metrics/train_brisque.py -m rgb_correl -c 1024 -e 4.4738315353908655
python src/colorization_metrics/metrics/train_brisque.py -m rgb_analysis -c 1024 -e 3.5378830258945904
python src/colorization_metrics/metrics/train_brisque.py -m rgb_all -c 1024 -e 1.751013270072053
