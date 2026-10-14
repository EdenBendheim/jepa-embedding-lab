"""Fixed-window aggregate diagnostics from a complete, local training journal."""
import hashlib
import json
import math
from statistics import mean


def summarize_training(report,*,window=100):
    if type(window) is not int or not 1<=window<=1000:raise ValueError('Window must be 1-1000 steps')
    steps=report.get('completed_steps');batch=report.get('settings',{}).get('batch_size');history=report.get('history')
    if (type(steps) is not int or not 1<=steps<=10000 or report.get('total_steps')!=steps
            or type(batch) is not int or not 1<=batch<=32 or not isinstance(history,list) or len(history)!=steps):
        raise ValueError('Summaries require the complete declared training budget')
    order=[]
    for index,row in enumerate(history,1):
        if (type(row.get('step')) is not int or row['step']!=index or not isinstance(row.get('train_indices'),list)
                or len(row['train_indices'])!=batch or len(set(row['train_indices']))!=batch
                or any(type(i) is not int or not 0<=i<50000 for i in row['train_indices'])):
            raise ValueError('Training history steps or actual image indices are invalid')
        order.append(row['train_indices'])
        try:
            numbers=[row['loss'],row['gradient_norm'],row['target']['std_mean'],row['prediction']['std_mean'],
                row['target']['covariance_off_diagonal_rms'],row['prediction']['covariance_off_diagonal_rms']]
            if any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in numbers):raise ValueError()
        except (KeyError,TypeError,ValueError) as exc:raise ValueError('Training diagnostics must be finite and nonnegative') from exc
    if (report.get('sample_draws')!=steps*batch or report.get('unique_training_images')!=len({i for row in order for i in row})
            or report.get('train_image_order_sha256')!=hashlib.sha256(json.dumps(order,separators=(',',':')).encode()).hexdigest()):
        raise ValueError('Actual training draw counts/order differ from the recorded journal')
    windows=[]
    for start in range(0,steps,window):
        rows=history[start:start+window]
        windows.append(dict(first_step=start+1,last_step=start+len(rows),count=len(rows),
            loss_mean=mean(row['loss'] for row in rows),loss_min=min(row['loss'] for row in rows),loss_max=max(row['loss'] for row in rows),
            gradient_norm_mean=mean(row['gradient_norm'] for row in rows),
            target_token_std_mean=mean(row['target']['std_mean'] for row in rows),
            prediction_token_std_mean=mean(row['prediction']['std_mean'] for row in rows),
            target_covariance_rms_mean=mean(row['target']['covariance_off_diagonal_rms'] for row in rows)))
    return dict(completed_steps=steps,window=window,sample_draws=report['sample_draws'],
        unique_training_images=report['unique_training_images'],windows=windows,
        limitation='Token spread and prediction loss are execution diagnostics, not downstream quality or proof against pooled-feature collapse')
