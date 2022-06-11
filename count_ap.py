import pandas as pd
import numpy as np

# average precision counting according to https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
def count_ap(trues, probs, id2label):
    df = pd.DataFrame({'prob': [], 'true': [], 'label': []})
    
    filtered = []
    for i in range(0,12):
        df_i = pd.DataFrame.from_dict({'prob': probs[i,:].cpu().detach().numpy(), 
                         'true': probs[i,:].cpu().detach().numpy(), 
                         'label': list(id2label.values())}
                       )
        df = df.append(df_i)
        del df_i
        
    ap_cur = []
    for cls in id2label.values():
        ap_cur.append(average_precision_score(y_true=df[df['label']==cls]['true'].values, y_score=df[df['label']==cls]['prob'].values))
        
    
    return np.mean(ap_cur)