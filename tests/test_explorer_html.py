from copy import deepcopy
import base64
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from jepa_lab.explorer_html import SCRIPT,render_html,write_html


def report():
    return dict(version=1,task='Frozen embedding neighborhood explorer',test_evaluated=False,
        restored_without_fitting=True,limitation='Descriptive only',training={},validation={},saved_index={},
        queries=[dict(index=10,label=0,class_name='airplane',neighbors=[dict(rank=1,index=0,label=0,class_name='airplane',cosine=1.)])])


class ExplorerHTMLTests(unittest.TestCase):
    def test_offline_document_escapes_payload_has_correct_script_hash_and_atomic_output(self):
        r=report();r['limitation']='</script><img src=x onerror=alert(1)>&'
        html=render_html(r)
        self.assertNotIn(r['limitation'],html);self.assertIn('\\u003c/script\\u003e',html)
        payload=html.split('<script id="report" type="application/json">')[1].split('</script>')[0]
        self.assertEqual(r,json.loads(payload))
        digest=base64.b64encode(hashlib.sha256(SCRIPT.encode()).digest()).decode()
        self.assertIn("script-src 'sha256-"+digest+"'",html)
        self.assertNotIn('innerHTML',SCRIPT);self.assertNotIn('fetch(',SCRIPT)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'view.html';write_html(path,r)
            self.assertEqual(html,path.read_text());self.assertFalse(path.with_suffix('.html.tmp').exists())
            changed=deepcopy(r);changed['queries'][0]['neighbors'][0]['cosine']=float('nan')
            with self.assertRaises(ValueError):write_html(path,changed)
            self.assertEqual(html,path.read_text())

    def test_rejects_test_reports_bad_rows_missing_provenance_and_unranked_neighbors(self):
        variants=[]
        for key,value in [('test_evaluated',True),('version',True),('queries',[]),('training',None)]:
            variants.append(report()|{key:value})
        for key,value in [('index',True),('label',10),('class_name','unknown')]:
            r=report();r['queries'][0][key]=value;variants.append(r)
        for key,value in [('rank',2),('cosine',float('inf')),('cosine',2.)]:
            r=report();r['queries'][0]['neighbors'][0][key]=value;variants.append(r)
        r=report();r['queries']*=2;variants.append(r)
        for r in variants:
            with self.assertRaises(ValueError):render_html(r)
