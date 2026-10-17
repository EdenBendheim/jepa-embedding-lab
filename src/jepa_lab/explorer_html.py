"""Self-contained, offline row-and-label explorer (no image/vector export)."""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path

SCRIPT = '''const data=JSON.parse(document.getElementById("report").textContent);
const select=document.getElementById("query");
const table=document.getElementById("neighbors");
function cell(row,value){const td=document.createElement("td");td.textContent=value;row.append(td);}
function show(){const q=data.queries[Number(select.value)];table.replaceChildren();
 document.getElementById("query-label").textContent="Validation row "+q.index+" · "+q.class_name;
 for(const n of q.neighbors){const tr=document.createElement("tr");
  for(const value of [n.rank,n.index,n.class_name,n.cosine.toFixed(6)])cell(tr,value);table.append(tr);}}
for(const [i,q] of data.queries.entries()){const option=document.createElement("option");
 option.value=i;option.textContent=q.index+" · "+q.class_name;select.append(option);}
select.addEventListener("change",show);
document.getElementById("limitations").textContent=data.limitation;
document.getElementById("provenance").textContent=JSON.stringify({training:data.training,validation:data.validation,saved_index:data.saved_index},null,2);
show();'''


def validate_report(report):
    if (not isinstance(report,dict) or type(report.get('version')) is not int or report['version']!=1
            or report.get('task')!='Frozen embedding neighborhood explorer'
            or report.get('test_evaluated') is not False or report.get('restored_without_fitting') is not True
            or not isinstance(report.get('limitation'),str)):
        raise ValueError('Expected a version-one frozen neighborhood report without test evaluation')
    queries=report.get('queries')
    if not isinstance(queries,list) or not 1<=len(queries)<=100:raise ValueError('Expected 1-100 queries')
    seen=set()
    for q in queries:
        if not isinstance(q,dict):raise ValueError('Invalid query')
        _identity(q)
        if q['index'] in seen:raise ValueError('Duplicate query')
        seen.add(q['index']);neighbors=q.get('neighbors')
        if not isinstance(neighbors,list) or not 1<=len(neighbors)<=100:raise ValueError('Expected 1-100 neighbors')
        ids=set();previous=float('inf')
        for rank,n in enumerate(neighbors,1):
            if not isinstance(n,dict):raise ValueError('Invalid neighbor')
            _identity(n);score=n.get('cosine')
            if (type(n.get('rank')) is not int or n['rank']!=rank or n['index'] in ids
                    or type(score) not in (int,float) or not math.isfinite(score)
                    or not -1.00001<=score<=1.00001 or score>previous):raise ValueError('Invalid ranked cosine neighbor')
            ids.add(n['index']);previous=score
    for field in ('training','validation','saved_index'):
        if not isinstance(report.get(field),dict):raise ValueError('Missing source provenance')
    json.dumps(report,allow_nan=False)
    return report


def _identity(row):
    from .cifar10 import CLASSES
    if (type(row.get('index')) is not int or not 0<=row['index']<50000
            or type(row.get('label')) is not int or not 0<=row['label']<10
            or row.get('class_name')!=CLASSES[row['label']]):raise ValueError('Invalid official row identity')


def render_html(report):
    validate_report(report)
    # JSON is inert data; escaping '<' prevents closing the script element.
    payload=json.dumps(report,allow_nan=False).replace('&','\\u0026').replace('<','\\u003c').replace('>','\\u003e')
    digest=base64.b64encode(hashlib.sha256(SCRIPT.encode()).digest()).decode()
    return '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'sha256-'''+digest+''''; base-uri 'none'; form-action 'none'">
<title>JEPA neighborhood explorer</title>
<style>body{max-width:960px;margin:40px auto;padding:0 24px;font:17px system-ui;color:#18202d;background:#fafbfd}h1{font-size:30px}select{font:inherit;padding:8px}table{width:100%;border-collapse:collapse;background:white}th,td{text-align:left;padding:12px;border-bottom:1px solid #d5dae1}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}p{line-height:1.6}.note{border-left:4px solid #4f6597;padding-left:16px;color:#3b4659}</style>
<h1>JEPA neighborhood explorer</h1>
<p>Inspect training neighbors of already observed validation examples. This view shows row identities and labels; raw images and feature vectors stay local and are not embedded.</p>
<label for="query">Validation example </label><select id="query"></select>
<h2 id="query-label"></h2><table><thead><tr><th>Rank</th><th>Training row</th><th>Class</th><th>Cosine</th></tr></thead><tbody id="neighbors"></tbody></table>
<p class="note" id="limitations"></p><details><summary>Source fingerprints and configuration</summary><pre id="provenance"></pre></details>
<noscript>Enable JavaScript to select queries. The companion JSON contains the same results.</noscript>
<script id="report" type="application/json">'''+payload+'''</script><script>'''+SCRIPT+'''</script></html>'''


def write_html(path,report):
    path=Path(path);content=render_html(report)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    try:temporary.write_text(content,encoding='utf-8');temporary.replace(path)
    finally:temporary.unlink(missing_ok=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('report',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    try:
        if args.report.stat().st_size>8*1024*1024:raise ValueError('Report exceeds 8 MiB')
        write_html(args.output,json.loads(args.report.read_text()))
    except (ValueError,TypeError,KeyError,OSError) as exc:p.error(str(exc))
    print(args.output)


if __name__=='__main__':main()
