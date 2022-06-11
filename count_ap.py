import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# average precision counting according to https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
def count_ap(trues, probs, id2label):
    
    df = pd.DataFrame({'prob': [], 'true': [], 'label': []})
    
    filtered = []
    for i in range(0,12):
        df_i = pd.DataFrame.from_dict({'prob': probs[i,:].cpu().detach().numpy(), 
                         'true': trues[i,:].cpu().detach().numpy(), 
                         'label': list(id2label.values())}
                       )
        df = df.append(df_i)
        del df_i
    # Count AP and mean
    ap_cur = []
    for cls in id2label.values():
        ap_cur.append(average_precision_score(y_true=df[df['label']==cls]['true'].values, y_score=df[df['label']==cls]['prob'].values))
        
    ap_cur = np.nan_to_num(ap_cur)
    return np.mean(ap_cur)