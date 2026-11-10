"""Touchable breakdowns: P&L by weekday, killzone and hour plus a month x year heatmap, with click-to-exclude what-if.

Everything is computed in the browser from a compact trade list (entry time + P&L), so excluding a slice recomputes the totals
instantly without re-running anything. Excluding a trade here only subtracts its recorded P&L: it does not re-simulate how later
trades would have sized or interacted, so the modal says so and points to a real re-run before you apply a change.
"""
from __future__ import annotations

import json
from html import escape

# Chicago-time killzones (the trade timestamps in a report are exchange/session time): name -> (start hour inclusive, end hour exclusive)
KILLZONES = {"Asia": (18, 24), "London": (2, 5), "NY open": (8, 11), "NY midday": (11, 14), "NY close": (14, 17)}


def compact_trades(trades) -> list[list]:
    """[[ISO entry time, pnl], ...] from Trade objects or dicts."""
    out = []
    for t in trades or []:
        ts = t.get("entry_time") if isinstance(t, dict) else getattr(t, "entry_time", None)
        pnl = t.get("pnl") if isinstance(t, dict) else getattr(t, "pnl", None)
        if ts is None or pnl is None:
            continue
        out.append([str(ts)[:16].replace(" ", "T"), round(float(pnl), 2)])
    return out


def breakdown_section(trades) -> str:
    data = compact_trades(trades)
    if not data:
        return '<p class="muted">No trades to break down.</p>'
    payload = json.dumps({"trades": data, "zones": KILLZONES}).replace("<", "\\u003c")
    return f"""
<div id="bd-root" data-payload='{escape(payload, quote=True)}'>
  <p>Tap a <b style="color:#f0596a">red</b> slice (a weekday, killzone or hour that lost money) to leave it out and see what the result would have been.</p>
  <div id="bd-totals" style="font-size:15px;margin:6px 0 12px;"></div>
  <div id="bd-charts"></div>
  <h4>Month by year</h4><div id="bd-heat" style="overflow-x:auto;"></div>
  <div id="bd-modal" hidden style="position:fixed;inset:0;background:rgba(0,0,0,.55);display:none;align-items:center;justify-content:center;z-index:50;">
    <div style="background:var(--card,#161b22);color:inherit;padding:18px 20px;border-radius:12px;max-width:420px;width:92%;">
      <h3 style="margin-top:0">What if you left this out?</h3><div id="bd-modal-body"></div>
      <p class="muted" style="font-size:12px">This only removes those trades' recorded P&amp;L. It does not re-simulate sizing or the trades that would have followed, so confirm any change with a fresh run.</p>
      <button type="button" id="bd-apply" class="t58-btn primary sm">Leave it out</button> <button type="button" id="bd-cancel" class="t58-btn ghost sm">Cancel</button>
    </div>
  </div>
</div>
<script>
(function(){{
  var root=document.getElementById('bd-root'); if(!root) return;
  var P=JSON.parse(root.getAttribute('data-payload')), T=P.trades.map(function(t){{var d=new Date(t[0]);return {{d:d,pnl:t[1],wd:d.getDay(),h:d.getHours(),m:d.getMonth(),y:d.getFullYear()}};}});
  var DAYS=['Sun','Mon','Tue','Wed','Thu','Fri','Sat'], MON=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  var excl={{}};                                    // "kind:key" -> true
  function zoneOf(h){{ for(var z in P.zones){{ var r=P.zones[z]; if(h>=r[0]&&h<r[1]) return z; }} return 'Other'; }}
  function keyOf(t,kind){{ return kind==='wd'?t.wd:kind==='hr'?t.h:zoneOf(t.h); }}
  function out(t){{ return excl['wd:'+t.wd]||excl['hr:'+t.h]||excl['kz:'+zoneOf(t.h)]; }}
  function sum(a){{ return a.reduce(function(s,t){{return s+t.pnl;}},0); }}
  function money(v){{ return (v<0?'-':'')+'$'+Math.abs(v).toLocaleString(undefined,{{maximumFractionDigits:0}}); }}
  function group(kind){{ var g={{}}; T.forEach(function(t){{ var k=keyOf(t,kind); (g[k]=g[k]||[]).push(t); }}); return g; }}
  function bars(title,kind,order,label){{
    var g=group(kind), keys=order||Object.keys(g), mx=Math.max.apply(null,keys.map(function(k){{return Math.abs(sum(g[k]||[]));}}).concat([1]));
    return '<h4>'+title+'</h4><div style="display:flex;gap:4px;align-items:flex-end;height:110px;">'+keys.map(function(k){{
      var v=sum(g[k]||[]), n=(g[k]||[]).length, off=excl[kind+':'+k], h=Math.max(4,Math.abs(v)/mx*90);
      var col=off?'#555':(v<0?'#f0596a':'#3ed685');
      return '<button type="button" data-kind="'+kind+'" data-key="'+k+'" title="'+label(k)+': '+money(v)+' over '+n+' trades" style="flex:1;min-width:14px;border:0;background:none;padding:0;cursor:'+(v<0&&!off?'pointer':'default')+';">'+
        '<div style="height:'+h+'px;background:'+col+';border-radius:3px;margin-top:'+(90-h)+'px;opacity:'+(off?.5:1)+'"></div><div style="font-size:10px;">'+label(k)+'</div></button>';
    }}).join('')+'</div>'; }}
  function render(){{
    var kept=T.filter(function(t){{return !out(t);}}), all=sum(T), now=sum(kept);
    document.getElementById('bd-totals').innerHTML='Result: <b>'+money(now)+'</b> from '+kept.length+' trades'+(kept.length!==T.length?' <span class="muted">(was '+money(all)+' from '+T.length+')</span>':'')+
      (Object.keys(excl).length?' <button type="button" id="bd-reset" class="t58-btn ghost sm">Put everything back</button>':'');
    document.getElementById('bd-charts').innerHTML=bars('By weekday','wd',[1,2,3,4,5],function(k){{return DAYS[k];}})+bars('By killzone','kz',Object.keys(P.zones).concat(['Other']),function(k){{return k;}})+
      bars('By hour','hr',Array.apply(null,Array(24)).map(function(_,i){{return i;}}),function(k){{return k;}});
    var years=Array.from(new Set(T.map(function(t){{return t.y;}}))).sort(), cell={{}}; kept.forEach(function(t){{var k=t.y+'-'+t.m;cell[k]=(cell[k]||0)+t.pnl;}});
    var mx=Math.max.apply(null,Object.keys(cell).map(function(k){{return Math.abs(cell[k]);}}).concat([1]));
    document.getElementById('bd-heat').innerHTML='<table style="border-collapse:collapse;font-size:11px;"><tr><th></th>'+MON.map(function(m){{return '<th style="padding:2px 6px">'+m+'</th>';}}).join('')+'</tr>'+years.map(function(y){{
      return '<tr><th style="padding:2px 6px">'+y+'</th>'+MON.map(function(_,m){{var v=cell[y+'-'+m]; if(v===undefined) return '<td style="background:#222;padding:6px;"></td>';
        var a=.15+.75*Math.abs(v)/mx; return '<td title="'+money(v)+'" style="padding:6px;text-align:center;background:'+(v<0?'rgba(240,89,106,'+a+')':'rgba(62,214,133,'+a+')')+'">'+Math.round(v/100)/10+'k</td>';}}).join('')+'</tr>';}}).join('')+'</table>';
  }}
  var pending=null;
  root.addEventListener('click',function(e){{
    var b=e.target.closest('button[data-kind]'); if(b){{ var k=b.dataset.kind+':'+b.dataset.key; if(excl[k]){{delete excl[k];render();return;}}
      var g=group(b.dataset.kind)[b.dataset.key]||[]; if(sum(g)>=0) return;                          // only losing slices are removable
      var before=sum(T.filter(function(t){{return !out(t);}})); excl[k]=true; var after=sum(T.filter(function(t){{return !out(t);}})); delete excl[k];
      var removed=g.filter(function(t){{return !out(t);}}).length; pending=k;
      document.getElementById('bd-modal-body').innerHTML='<p>Without this slice: result goes from <b>'+money(before)+'</b> to <b>'+money(after)+'</b> ('+(before?((after-before)/Math.abs(before)*100).toFixed(1):'0')+'%), and <b>'+removed+'</b> trades are removed.</p>';
      var m=document.getElementById('bd-modal'); m.hidden=false; m.style.display='flex'; return; }}
    if(e.target.id==='bd-reset'){{ excl={{}}; render(); }}
    if(e.target.id==='bd-apply'&&pending){{ excl[pending]=true; pending=null; document.getElementById('bd-modal').style.display='none'; render(); }}
    if(e.target.id==='bd-cancel'){{ pending=null; document.getElementById('bd-modal').style.display='none'; }}
  }});
  render();
}})();
</script>"""
