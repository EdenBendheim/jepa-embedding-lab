from contextlib import redirect_stdout,redirect_stderr
import io,json
from pathlib import Path
import tempfile,unittest
from unittest.mock import patch
import torch

from jepa_lab.cifar10 import make_manifest
from jepa_lab.cifar10_train import main,train_cifar10
from jepa_lab.model import ModelConfig
from test_cifar10_train import Fixture


class ModelCLITests(unittest.TestCase):
    def test_custom_architecture_cli_resumes_exactly_and_rejects_changed_width(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);m=root/'manifest.json';m.write_text(json.dumps(make_manifest(Fixture(root),Fixture(root,train=False))))
            checkpoint=root/'model.pt';args=['train',str(root),'--manifest',str(m),'--steps','2','--batch-size','2',
                '--embedding-dim','64','--heads','8','--encoder-depth','1','--predictor-depth','1','--checkpoint',str(checkpoint)]
            with patch('jepa_lab.cifar10_train.CIFAR10Binary',Fixture),redirect_stdout(io.StringIO()):
                with patch('sys.argv',args):main()
                with patch('sys.argv',args+['--resume',str(checkpoint)]):main()
                before=checkpoint.read_bytes();Fixture.accessed.clear()
                bad=args+['--resume',str(checkpoint),'--embedding-dim','32']
                with patch('sys.argv',bad),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):main()
                self.assertEqual([],Fixture.accessed);self.assertEqual(before,checkpoint.read_bytes())
                expected=train_cifar10(root,m,steps=4,batch_size=2,config=ModelConfig(patch_dim=48,embedding_dim=64,heads=8,encoder_depth=1))
            state=torch.load(checkpoint,weights_only=True)
            self.assertEqual(expected['history'],state['training_history']);self.assertEqual(8,state['config']['heads'])
