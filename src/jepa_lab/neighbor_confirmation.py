"""Score a fixed selected neighbor index on reserved confirmation examples."""
from .confirmation import verify_confirmation_sources
from .neighbor_artifacts import load_neighbors


def score_neighbor_confirmation(path,train,tuning,fresh,selection):
    context,excluded=verify_confirmation_sources(train,tuning,fresh,selection)
    index,original=load_neighbors(path,train,tuning)
    return dict(version=1,task='Frozen neighbor confirmation',original_index=original,
        manifest_sha256=context['manifest_sha256'],confirmation_selection_sha256=fresh.selection_sha256,
        confirmation=index.score(fresh.features,fresh.labels),confirmation_features=fresh.summary(),
        excluded_validation_count=len(excluded),normalization_refitted=False,index_refitted=False,
        settings_selected=False,test_evaluated=False,
        limitation='Untuned rows relative to declared sources; not official test accuracy. Reusing scores for later selection removes their confirmation role.')
