from copy import deepcopy
import hashlib
import json
import unittest

from jepa_lab.training_diagnostics import summarize_training


def report():
    order=[[i,i+6] for i in range(6)]
    return dict(completed_steps=6,total_steps=6,settings=dict(batch_size=2),sample_draws=12,unique_training_images=12,
        train_image_order_sha256=hashlib.sha256(json.dumps(order,separators=(',',':')).encode()).hexdigest(),
        history=[dict(step=i+1,train_indices=order[i],loss=float(i),gradient_norm=1.,
            target=dict(std_mean=.2,covariance_off_diagonal_rms=.1),prediction=dict(std_mean=.3,covariance_off_diagonal_rms=.1)) for i in range(6)])


class TrainingDiagnosticsTests(unittest.TestCase):
    def test_windows_include_partial_tail_and_do_not_modify_source(self):
        data=report();before=deepcopy(data);summary=summarize_training(data,window=4)
        self.assertEqual([4,2],[r['count'] for r in summary['windows']]);self.assertEqual([1.5,4.5],[r['loss_mean'] for r in summary['windows']])
        self.assertEqual(6,summary['windows'][-1]['last_step']);self.assertEqual(before,data)

    def test_missing_steps_nonfinite_diagnostics_changed_order_and_bad_windows_fail(self):
        for change in ('missing','step','loss','spread','order','count'):
            data=report()
            if change=='missing':data['history'].pop()
            if change=='step':data['history'][0]['step']=True
            if change=='loss':data['history'][0]['loss']=float('nan')
            if change=='spread':data['history'][0]['target']['std_mean']=-.1
            if change=='order':data['history'][0]['train_indices'].reverse()
            if change=='count':data['unique_training_images']+=1
            with self.subTest(change=change),self.assertRaises(ValueError):summarize_training(data)
        for window in (True,0,1001):
            with self.assertRaises(ValueError):summarize_training(report(),window=window)
