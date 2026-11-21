"""Compare regional readout gains with matched random and RGB controls."""

from statistics import mean

from experiments.multiblock_spatial_scoring import _pair


TRAINED = ('separate', 'joint_matched', 'joint_extended')


def compare_readouts(labels, predictions, seeds):
    def prediction(source, readout, view='original'):
        return predictions[f'{source}/{readout}'][view]

    def sensitivity(source, view):
        original = prediction(source, 'quadrants')
        perturbed = prediction(source, 'quadrants', view)
        return _pair(labels, original, perturbed)['accuracy_delta']

    def region_gain(source):
        return _pair(labels, prediction(source, 'mean'), prediction(source, 'quadrants'))

    paired = {}
    region_rgb = prediction('rgb', 'quadrants')
    for seed in seeds:
        random_source = f'{seed}/random'
        random_regions = prediction(random_source, 'quadrants')
        random_gain = region_gain(random_source)
        for arm in TRAINED:
            source = f'{seed}/{arm}'
            regions = prediction(source, 'quadrants')
            gain = region_gain(source)
            paired[source] = dict(
                quadrant_gain=gain,
                quadrant_vs_random=_pair(labels, random_regions, regions),
                quadrant_vs_regional_rgb=_pair(labels, region_rgb, regions),
                quadrant_gain_minus_random=gain['accuracy_delta']-random_gain['accuracy_delta'],
                shuffle_delta_minus_random={
                    view: sensitivity(source, view)-sensitivity(random_source, view)
                    for view in ('patch_shuffle', 'quadrant_shuffle')
                })
        paired[random_source] = dict(quadrant_gain=random_gain)
    paired['rgb'] = dict(quadrant_gain=region_gain('rgb'))

    def average(sources):
        return {
            readout: {
                view: mean(float((prediction(source, readout, view) == labels).sum())/len(labels)
                           for source in sources)
                for view in predictions[f'{sources[0]}/{readout}']
            }
            for readout in ('mean', 'repeated_mean', 'quadrants')
        }

    means = {arm: average([f'{seed}/{arm}' for seed in seeds]) for arm in (*TRAINED, 'random')}
    means['rgb'] = average(['rgb'])
    repeats = {source: int((prediction(source, 'mean') != prediction(source, 'repeated_mean')).sum())
               for source in paired}
    return dict(paired=paired, mean_accuracy=means, repeated_mean_prediction_disagreements=repeats)
