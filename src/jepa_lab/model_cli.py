"""Shared explicit RGB architecture options for local training entry points."""
from .model import ModelConfig
from .model_size import validate_local_config


def add_model_options(parser):
    group=parser.add_argument_group('RGB model architecture (must match on resume)')
    group.add_argument('--position-encoding',choices=('learned','sincos2d'),default='learned')
    group.add_argument('--embedding-dim',type=int,default=32)
    group.add_argument('--heads',type=int,default=4)
    group.add_argument('--encoder-depth',type=int,default=2)
    group.add_argument('--predictor-depth',type=int,default=1)


def config_from_args(args):
    return validate_local_config(ModelConfig(patch_dim=48,embedding_dim=args.embedding_dim,heads=args.heads,
        encoder_depth=args.encoder_depth,predictor_depth=args.predictor_depth,position_encoding=args.position_encoding))
