from functions.higgs.h01_preprocess import *
from functions.higgs.h02_triggers import *
from functions.higgs.h03_diffusion import *
from functions.higgs.h04_baseline import *

hp = HiggsPreprocess()
hp.run()

ht = HiggsTriggers()
ht.run()
ht.slearner(rerun=True)
ht.slearner_tree()

hd = HiggsDiffusion()
hd.run()
hd.slearner()
hd.slearner_tree()

hb = HiggsBaselines(save_results=False, skip_non_lr=False)
hb.run()
