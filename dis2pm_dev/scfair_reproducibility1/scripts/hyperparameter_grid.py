import pandas as pd
import os 
#import itertools

cf_weight_max = 1000
clf_weight_max = 1000
adv_clf_weight_max = 1000
nsteps = 10

cf_common_ratio = cf_weight_max**(1/nsteps)
clf_common_ratio = clf_weight_max**(1/nsteps)
adv_clf_common_ratio = adv_clf_weight_max**(1/nsteps)

cf_weight = [0]+[round(cf_common_ratio**i) for i in range(nsteps+1)]
clf_weight = [0]+[round(clf_common_ratio**i) for i in range(nsteps+1)]
adv_clf_weight = [0]+[round(adv_clf_common_ratio**i) for i in range(nsteps+1)]

hyper = [ [i,j,k] for i in cf_weight for j in clf_weight for k in adv_clf_weight] 

df_hyper = pd.DataFrame(
              hyper,
              columns=['cf_weight', 'clf_weight', 'adv_clf_weight']
            )
         
os.makedirs('data/output/hyperparam_grid', exist_ok=True)  
df_hyper.to_csv('data/output/hyperparam_grid/hyperparam_grid.csv')  



