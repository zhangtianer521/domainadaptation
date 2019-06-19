"""
needs GPUtil
"""
import sys
sys.path.append('/home/wzhang/Self_Ensembling/')

import GPUtil
import json
import main
import time
from threading import Thread

gpus = GPUtil.getGPUs()
gpus_available = [0,1,2,3]

"""
different experiments, each one is a dictionary with the changes performed at the base json
"""
experiments = [
    # {
    #     "experiment_name": "logs/adapt3_dice",
    #     "adapt_centers": [3],
    #     "val_centers": [3,4],
    #     "consistency_loss": "dice_loss",
    # },
    {
        "experiment_name": "logs/adapt3_mse1",
        "adapt_centers": [3],
        "val_centers": [3,4],
        "consistency_loss": "mse",
    },
    # {
    #     "experiment_name": "logs/adapt3_mse2",
    #     "adapt_centers": [3],
    #     "val_centers": [3,4],
    #     "consistency_loss": "mse",
    # },
    # {
    #     "experiment_name": "logs/adapt3_mse3",
    #     "adapt_centers": [3],
    #     "val_centers": [3,4],
    #     "consistency_loss": "mse",
    # },
    # {
    #     "experiment_name": "logs/adapt3_cross_entropy",
    #     "adapt_centers": [3],
    #     "val_centers": [3,4],
    #     "consistency_loss": "cross_entropy",
    # },
    # {
    #     "experiment_name": "logs/ema0.9_emalatee0.99",
    #     "ema_alpha": 0.9,
    #     "ema_alpha_late": 0.99,
    #     "ema_late_epoch": 50,
    # },
    # {
    #     "experiment_name": "logs/emaepoch100",
    #     "ema_late_epoch": 100,
    # },
]

threads = {}
for i in range(0, len(gpus)):
    threads[str(i)] = None

for exp in experiments:
    print(threads)
    with open('../experiments/domain_adaptation.json') as f:
        base_experiment = json.load(f)
        for k in exp:
            base_experiment[k] = exp[k]
    done = False
    # iter = 0
    while True:
        gpus = GPUtil.getGPUs()
        # iter += 10
        # print("Running "+iter+" .... ")
        for i, gpu in enumerate(gpus):
            if (int(gpu.memoryFree) > 8000) and (threads[str(i)] is None) and i in gpus_available:
                done = True
                base_experiment["gpu"] = str(i)
                thread = Thread(target=main.run_main, args=(base_experiment,))
                print("Starting new thread for experiment: %s in gpu %s" % (exp, str(i)))
                threads[str(i)] = thread
                thread.start()
                break
        for k in threads:
            if threads[k] is not None and not threads[k].is_alive():
                threads[k] = None

        if done:
            break
        time.sleep(60)

for t in threads:
    if t is not None:
        t.join()
