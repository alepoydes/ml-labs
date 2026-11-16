"""Offline inspection using the same CSV records and canonical PNG sheets."""

import json
from pathlib import Path

from coursekit.labs.visual_queries_data import (
    EXAMPLE_FIELDS,
    metadata,
    rows,
    sheet_path,
    split_label,
)


def build(root):
    root = Path(root)
    meta = metadata(root)
    splits = {}
    order = {
        f"adaptation/{name}/{part}": 2 * i + j
        for i, name in enumerate(meta["episodes"])
        for j, part in enumerate(("support", "query"))
    }
    for split in sorted(
        meta["splits"], key=lambda name: (not name.startswith("base/"), order.get(name, -1))
    ):
        folder = root / split
        splits[split] = [
            [
                i,
                [f"{split}/{sheet_path(int(r['sheet']))}", int(r["row"]), int(r["column"])],
                r["question"],
                r["answer"],
            ]
            for i, r in enumerate(rows(folder / "examples.csv", EXAMPLE_FIELDS), 1)
        ]
    payload = json.dumps(
        dict(
            shape=meta["image_shape"],
            splits=splits,
            labels={s: split_label(meta, s) for s in splits},
            sizes=dict(
                sorted(
                    meta.get("observable_rules", {}).get("size_pixels", {}).items(),
                    key=lambda item: item[1][0],
                )
            ),
        )
    ).replace("<", "\\u003c")
    html = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Visual Queries · Practice data</title><style>
body{font:17px system-ui;margin:24px auto;max-width:1100px;padding:0 20px;color:#203047;background:#f5f7fa}
h1{margin-bottom:.3em}button,select,input{font:inherit;padding:6px;margin:4px}main{display:flex;gap:28px;flex-wrap:wrap;margin-top:20px}
canvas{width:100%;height:auto;background:white;border:1px solid #ccd4df;image-rendering:pixelated}.visual{width:min(512px,100%)}#sizes{width:100%;height:auto}article{flex:1;min-width:260px}#question{font-size:24px;line-height:1.5}.token{background:#e5edf9;border-radius:4px;padding:2px 4px}small{color:#52647a;overflow-wrap:anywhere}article{overflow-wrap:anywhere}select{max-width:100%}#answer{font-size:24px;min-height:1.5em}footer{margin-top:24px}</style>
<h1>Visual Queries</h1><p>Choose a split and inspect its image, question and answer. Text is for inspection; the learning API supplies only numeric tokens and pixels.</p>
<label>Split <select id="split"></select></label><br><button id="previous">Previous</button><button id="next">Next</button><label>Question <input id="number" type="number" min="1" value="1" style="width:7em"></label><span id="count"></span>
<main><section class="visual"><canvas id="image"></canvas><details><summary>Symbol size reference</summary><small>Maximum ink-box dimension; lines use the same display scale as the image.</small><svg id="sizes" aria-label="Symbol size reference"></svg></details></section><article><p id="question"></p><button id="reveal">Show answer</button><p id="answer"></p><small id="identity"></small><p id="error" role="alert"></p></article></main>
<footer><small>One-based PNG sheet coordinates. Questions are independent. Use support answers for adaptation; query answers are for local evaluation.</small><p><a href="SOURCES.md">Source attribution</a></p></footer>
<script id="dataset" type="application/json">PAYLOAD</script>
<script>const data=JSON.parse(document.getElementById('dataset').textContent);
const $=id=>document.getElementById(id); const selector=$('split');
const ns='http://www.w3.org/2000/svg', guide=$('sizes');guide.setAttribute('viewBox',`0 0 ${data.shape[1]} ${Object.keys(data.sizes).length*30}`);
Object.entries(data.sizes).forEach(([name,[lo,hi]],i)=>{const y=20+i*30,line=document.createElementNS(ns,'line'),label=document.createElementNS(ns,'text');line.setAttribute('x1',8);line.setAttribute('x2',8+lo);line.setAttribute('y1',y);line.setAttribute('y2',y);line.setAttribute('stroke','#203047');line.setAttribute('stroke-width',4);label.setAttribute('x',150);label.setAttribute('y',y+5);label.textContent=`${name}: ${lo}–${hi} px`;guide.append(line,label)});
for(const name of Object.keys(data.splits)){const o=document.createElement('option');o.value=name;o.textContent=data.labels[name];selector.append(o)}
let index=0, reveal=false, revision=0;
function draw(){$('error').textContent='';const rev=++revision; const records=data.splits[selector.value];index=Math.max(0,Math.min(index,records.length-1)); const [id,tile,words,answer]=records[index];
$('number').value=index+1;$('number').max=records.length;$('count').textContent=`of ${records.length}`;$('previous').disabled=index===0;$('next').disabled=index===records.length-1;
$('question').textContent=words;
$('answer').textContent=reveal?answer:'';$('reveal').textContent=reveal?'Hide answer':'Show answer';
$('identity').textContent=`Question ${id} · ${data.labels[selector.value]} · ${tile[0].split('/').at(-1)} · row ${tile[1]}, column ${tile[2]}`;$('identity').title=tile[0];
const canvas=$('image');canvas.width=data.shape[1];canvas.height=data.shape[0];const ctx=canvas.getContext('2d');ctx.imageSmoothingEnabled=false;ctx.fillStyle='white';ctx.fillRect(0,0,canvas.width,canvas.height);
const image=new Image();image.onload=()=>{if(rev===revision)ctx.drawImage(image,(tile[2]-1)*canvas.width,(tile[1]-1)*canvas.height,canvas.width,canvas.height,0,0,canvas.width,canvas.height)};image.onerror=()=>{$('error').textContent='Cannot open the canonical PNG sheet. Check that the resource directory is complete.'};image.src=tile[0];}
selector.onchange=()=>{index=0;reveal=false;draw()};$('previous').onclick=()=>{index--;reveal=false;draw()};$('next').onclick=()=>{index++;reveal=false;draw()};$('number').onchange=()=>{index=Number($('number').value)-1;reveal=false;draw()};$('reveal').onclick=()=>{reveal=!reveal;draw()};draw();</script></html>"""
    (root / "index.html").write_text(html.replace("PAYLOAD", payload), encoding="utf-8")
    return root / "index.html"
