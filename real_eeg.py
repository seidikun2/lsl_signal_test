"""Record an existing EEG outlet and compare it with the same stream in XDF."""
import csv
import io
import json
import math
import time
from pathlib import Path
from xml.etree import ElementTree

import numpy as np


def stream_metadata(info):
    xml=info.as_xml()
    channels=ElementTree.fromstring(xml).findall('./desc/channels/channel')
    labels=[c.findtext('label','') for c in channels]
    units=[c.findtext('unit','') for c in channels]
    if len(labels)!=info.channel_count():labels=[];units=[]
    return dict(name=info.name(),type=info.type(),source_id=info.source_id(),uid=info.uid(),
                channels=info.channel_count(),rate_hz=info.nominal_srate(),
                channel_format=info.channel_format(),labels=labels,units=units,xml=xml)


def list_streams():
    from pylsl import resolve_streams
    streams=resolve_streams(wait_time=2.)
    for info in streams:
        print(f'{info.name()} | type={info.type()} | {info.channel_count()} channels | '
              f'{info.nominal_srate():g} Hz | source_id={info.source_id()}')
    if not streams:print('No LSL streams found. Start the EEG LSL outlet first.')


def settings(document):
    cfg=dict(document['real_eeg'])
    if not isinstance(cfg.get('stream_name'),str) or not cfg['stream_name']:
        raise ValueError('real_eeg.stream_name must identify the EEG outlet')
    if not isinstance(cfg.get('marker_name',''),str):raise ValueError('marker_name must be a string')
    for key in ('duration_s','buffer_s','read_chunk','receiver_poll_hz','resolve_timeout_s','clock_update_s'):
        if not math.isfinite(cfg[key]) or cfg[key]<=0:raise ValueError(f'real_eeg.{key} must be positive and finite')
    for key in ('buffer_s','read_chunk'):
        if int(cfg[key])!=cfg[key]:raise ValueError(f'real_eeg.{key} must be an integer')
        cfg[key]=int(cfg[key])
    return cfg


def _fresh_root(output):
    root=Path(output)
    if root.exists() and any(root.iterdir()):raise ValueError(f'Output must be empty: {root}. Use --out with a new directory.')
    root.mkdir(parents=True,exist_ok=True)
    return root


def capture(cfg,output='results-real'):
    """Consume the actual outlet without adding channels, filtering or resampling."""
    from pylsl import StreamInlet,resolve_byprop,resolve_streams,local_clock
    cfg=settings({'real_eeg':cfg})
    # Check the destination before opening hardware streams; create it only after
    # successful connection, so a failed discovery leaves no result folder.
    root=Path(output)
    if root.exists() and any(root.iterdir()):raise ValueError('Output must be empty; choose a new --out directory')
    def find(prop,value,timeout):
        found=resolve_byprop(prop,value,timeout=timeout)
        if len(found)>1:raise ValueError(f'Multiple streams match {value!r}; use a unique name (see --list-streams)')
        if not found:raise RuntimeError(f'LSL stream not found: {value}. Start the device outlet first.')
        return found[0]
    print('[LSL] Looking for EEG: '+cfg['stream_name'],flush=True)
    selected={'eeg':find('name',cfg['stream_name'],cfg['resolve_timeout_s'])}
    marker_name=cfg.get('marker_name','')
    if marker_name:marker=find('name',marker_name,cfg['resolve_timeout_s'])
    else:
        candidates=[i for i in resolve_streams(wait_time=min(2.,cfg['resolve_timeout_s'])) if i.type().lower() in ('marker','markers')]
        if len(candidates)>1:raise ValueError('Multiple marker streams found; select one with --marker-name (see --list-streams)')
        marker=candidates[0] if candidates else None
    if marker and marker.uid()==selected['eeg'].uid():raise ValueError('EEG and marker selections refer to the same outlet')
    if marker:selected['marker']=marker
    if selected['eeg'].channel_format()==3:raise ValueError('EEG outlet must contain numeric samples')
    inlets={};metadata={};states={};observations=[]
    status='completed';error='';started=None
    try:
        for role,info in selected.items():
            inlet=StreamInlet(info,max_buflen=cfg['buffer_s'],max_chunklen=0,recover=False,processing_flags=0)
            inlets[role]=inlet
            inlet.open_stream(timeout=cfg['resolve_timeout_s'])
            offset=inlet.time_correction(timeout=cfg['resolve_timeout_s'])
            metadata[role]=stream_metadata(inlet.info(timeout=cfg['resolve_timeout_s']))
            states[role]=dict(values=[],raw=[],corrected=[],arrival=[],correction=[],current=offset)
            observations.append(dict(stream=role,measured=local_clock(),offset=offset))
        # The clock setup can take seconds. Begin a fresh observation window,
        # rather than reporting the setup backlog as acquisition latency.
        flushed={role:inlet.flush() for role,inlet in inlets.items()}
        started=local_clock();finished=started
        root=_fresh_root(output)
        print(f'[LSL] Connected: {metadata["eeg"]["channels"]} channels, {metadata["eeg"]["rate_hz"]:g} Hz. '
              +('Markers: '+metadata['marker']['name'] if marker else 'No marker stream found; recording EEG only.'),flush=True)
        print(f'[LSL] Recording for {cfg["duration_s"]:g} s. Ctrl+C saves a shorter recording.',flush=True)
        period=1/cfg['receiver_poll_hz'];next_read=started;refresh=started+cfg['clock_update_s'];progress=started+10
        try:
            while local_clock()<started+cfg['duration_s']:
                now=local_clock()
                if now>=refresh:
                    for role,inlet in inlets.items():
                        try:
                            offset=inlet.time_correction(timeout=.001)
                            states[role]['current']=offset
                            observations.append(dict(stream=role,measured=local_clock(),offset=offset))
                        except TimeoutError:pass
                    refresh=local_clock()+cfg['clock_update_s']
                for role,inlet in inlets.items():
                    values,stamps=inlet.pull_chunk(timeout=0.,max_samples=cfg['read_chunk'])
                    read=local_clock()
                    if not stamps:continue
                    state=states[role];raw=np.asarray(stamps,dtype=np.float64)
                    state['values'].append(np.asarray(values,dtype=str if role=='marker' else np.float64))
                    state['raw'].append(raw);state['corrected'].append(raw+state['current'])
                    state['arrival'].append(np.full(len(raw),read));state['correction'].append(np.full(len(raw),state['current']))
                now=local_clock()
                if now>=progress:
                    print(f'[LSL] {now-started:.0f} s; {sum(len(v) for v in states["eeg"]["raw"])} EEG samples saved in memory.',flush=True)
                    progress=now+10
                next_read+=period*max(1,int((now-next_read)//period)+1)
                time.sleep(max(0.,next_read-now))
        except KeyboardInterrupt:status='stopped by user'
        except Exception as exc:status='interrupted';error=str(exc)
        finally:finished=local_clock()
    finally:
        for inlet in inlets.values():
            try:inlet.close_stream()
            except Exception:pass  # Closing must not prevent saving captured data.
    # Setup failures propagate; a recording that started always saves its data.
    arrays={}
    for role,state in states.items():
        for field in ('values','raw','corrected','arrival','correction'):
            arrays[role+'_'+field]=(np.concatenate(state[field]) if state[field] else
                np.empty((0,metadata[role]['channels']),dtype=str if role=='marker' else float) if field=='values' else np.empty(0))
    meta=dict(mode='real LSL capture',settings=cfg,streams=metadata,start=started,end=finished,
              status=status,error=error,clock_observations=observations,flushed_before_start=flushed,
              marker_status='recorded' if marker else 'no marker stream found')
    arrays['metadata_json']=np.array(json.dumps(meta,allow_nan=False))
    np.savez_compressed(root/'recordings.npz',**arrays)
    report(root,meta,arrays)
    print('Results: '+str(root/'summary.html'),flush=True)
    if error:raise RuntimeError('Recording interrupted; available data were saved. '+error)
    if not len(arrays['eeg_raw']):raise RuntimeError('No EEG samples received; the empty recording and report were saved')
    return root/'recordings.npz'


def match_samples(captured,reference,tolerance=1e-6,allow_equal=False):
    """Match original source stamps one-to-one, only inside shared coverage."""
    captured=np.asarray(captured,dtype=float);reference=np.asarray(reference,dtype=float)
    if not math.isfinite(tolerance) or tolerance<=0:raise ValueError('Timestamp tolerance must be positive')
    for stamps in (captured,reference):
        if not np.isfinite(stamps).all() or (np.diff(stamps)<0 if allow_equal else np.diff(stamps)<=0).any():
            raise ValueError('Original timestamps must be strictly increasing; duplicated stamps or clock resets need inspection')
    if not len(captured) or not len(reference):raise ValueError('One recording has no samples to compare')
    low=max(captured[0],reference[0]);high=min(captured[-1],reference[-1])
    if low>high+tolerance:raise ValueError('No shared time range. Record Python and the reference simultaneously.')
    ci=np.flatnonzero((captured>=low-tolerance)&(captured<=high+tolerance))
    ri=np.flatnonzero((reference>=low-tolerance)&(reference<=high+tolerance))
    if allow_equal:
        pairs=[];i=j=0
        while i<len(ci) and j<len(ri):
            delta=captured[ci[i]]-reference[ri[j]]
            if abs(delta)<=tolerance:pairs.append((ci[i],ri[j]));i+=1;j+=1
            elif delta<0:i+=1
            else:j+=1
        a=np.array([p[0] for p in pairs],dtype=int);b=np.array([p[1] for p in pairs],dtype=int)
    else:
        candidates=np.searchsorted(reference,captured[ci])
        left=np.clip(candidates-1,0,len(reference)-1);right=np.clip(candidates,0,len(reference)-1)
        nearest=np.where(np.abs(reference[left]-captured[ci])<=np.abs(reference[right]-captured[ci]),left,right)
        valid=np.abs(reference[nearest]-captured[ci])<=tolerance
        a,b=ci[valid],nearest[valid]
    if len(np.unique(b))!=len(b):raise ValueError('Ambiguous timestamp matches; recordings need inspection')
    return dict(captured=a,reference=b,shared_start=float(low),shared_end=float(high),
                captured_in_range=len(ci),reference_in_range=len(ri),
                unmatched_captured=len(ci)-len(a),unmatched_reference=len(ri)-len(b))


def _first(value,default=''):
    return value[0] if isinstance(value,list) and value else value if value is not None else default


def _xdf_stream(stream,expected):
    info=stream['info']
    name=_first(info.get('name'))
    source_id=_first(info.get('source_id'))
    if name!=expected['name']:return False
    return not expected.get('source_id') or not source_id or expected['source_id']==source_id


def load_reference(path,meta):
    """Use unmodified source clocks, not a smoothed XDF time vector."""
    path=Path(path)
    if path.suffix.lower() not in ('.xdf','.xdfz'):
        raise ValueError('Reference must be a simultaneous LabRecorder XDF file with the original LSL timestamps.')
    try:import pyxdf
    except ImportError as exc:raise RuntimeError('Install the optional XDF reader in the same Python: python -m pip install pyxdf') from exc
    streams,_=pyxdf.load_xdf(str(path),select_streams=[{'name':m['name']} for m in meta['streams'].values()],
        synchronize_clocks=False,dejitter_timestamps=False,handle_clock_resets=False,verbose=False)
    result={}
    for role,expected in meta['streams'].items():
        matches=[s for s in streams if _xdf_stream(s,expected)]
        exact_uid=[s for s in matches if expected.get('uid') and _first(s['info'].get('uid'))==expected['uid']]
        if expected.get('uid') and any(_first(s['info'].get('uid')) for s in matches):matches=exact_uid
        if not matches and role=='marker':continue
        if len(matches)!=1:raise ValueError(f'Expected exactly one matching {role} stream in XDF; found {len(matches)}')
        stream=matches[0]
        desc=_first(stream['info'].get('desc'),{})
        channels=_first(desc.get('channels'),{}) if isinstance(desc,dict) else {}
        channel_info=channels.get('channel',[]) if isinstance(channels,dict) else []
        result[role]=dict(raw=np.asarray(stream['time_stamps'],dtype=float),
            values=np.asarray(stream['time_series'],dtype=str if role=='marker' else float),
            labels=[_first(c.get('label')) for c in channel_info],units=[_first(c.get('unit')) for c in channel_info])
    return result,'XDF: original timestamps, no smoothing or clock adjustment'


def compare_values(values,reference,paired,labels):
    if values.ndim!=2 or reference.ndim!=2 or values.shape[1]!=reference.shape[1]:
        raise ValueError('Channel counts differ; do not compare before checking channel mapping')
    a,b=paired['captured'],paired['reference']
    if not len(a):raise ValueError('No original timestamps matched in the shared time range')
    left,right=values[a],reference[b]
    result=[]
    for i in range(values.shape[1]):
        valid=np.isfinite(left[:,i])&np.isfinite(right[:,i])
        diff=left[valid,i]-right[valid,i]
        equal=(left[:,i]==right[:,i])|(np.isnan(left[:,i])&np.isnan(right[:,i]))
        result.append(dict(channel=labels[i],matched=len(a),different_values=int(np.count_nonzero(~equal)),
            nonfinite_pairs=int(np.count_nonzero(~valid)),max_difference=float(np.max(np.abs(diff))) if len(diff) else None))
    return result


def compare(recording,reference,output=None):
    import shutil
    recording=Path(recording)
    with np.load(recording,allow_pickle=False) as saved:arrays=dict(saved)
    meta=json.loads(str(arrays['metadata_json']))
    if meta.get('mode')!='real LSL capture':raise ValueError('Use a recording from --real-eeg')
    refs,description=load_reference(reference,meta)
    expected=meta['streams']['eeg'];actual=refs['eeg']
    labels=expected.get('labels',[]);ref_labels=actual['labels']
    labeled=bool(labels and all(labels) and ref_labels and all(ref_labels))
    known_labels=[(a,b) for a,b in zip(labels,ref_labels) if a and b]
    if any(a!=b for a,b in known_labels):raise ValueError('Channel labels/order differ between Python and reference; no automatic reordering was applied')
    labeled=labeled and len(labels)==len(ref_labels)==expected['channels']
    units=expected.get('units',[]);ref_units=actual.get('units',[])
    if units and ref_units and any(a and b and a!=b for a,b in zip(units,ref_units)):
        raise ValueError('Channel units differ; no automatic scale change was applied')
    labels=[value or f'Channel {i+1}' for i,value in enumerate(labels)] if labels else [f'Channel {i+1}' for i in range(expected['channels'])]
    paired=match_samples(arrays['eeg_raw'],actual['raw'])
    rows=compare_values(arrays['eeg_values'],actual['values'],paired,labels)
    comparison=dict(reference=str(Path(reference).resolve()),description=description,paired=paired,
        rows=rows,channel_labels_verified=labeled,known_labels_checked=len(known_labels),
        timestamp_difference_ms=(arrays['eeg_raw'][paired['captured']]-actual['raw'][paired['reference']])*1000)
    if 'marker' in meta['streams'] and 'marker' in refs:
        try:
            marker_pair=match_samples(arrays['marker_raw'],refs['marker']['raw'],allow_equal=True)
            a,b=marker_pair['captured'],marker_pair['reference']
            if arrays['marker_values'].shape[1:]!=refs['marker']['values'].shape[1:]:raise ValueError('Marker channel counts differ')
            marker_pair['different_values']=int(np.count_nonzero(np.any(arrays['marker_values'][a]!=refs['marker']['values'][b],axis=1)))
            marker_pair['timestamp_difference_ms']=(arrays['marker_raw'][a]-refs['marker']['raw'][b])*1000
            comparison['markers']=marker_pair
        except ValueError as exc:comparison['marker_error']=str(exc)
    else:comparison['marker_error']='Markers not present in both recordings; no marker comparison was made.'
    root=_fresh_root(output) if output else recording.parent
    if output:shutil.copy2(recording,root/'recordings.npz')
    report(root,meta,arrays,comparison,refs)
    print('Comparison: '+str(root/'summary.html'))
    return comparison


def _csv(path,rows,fields=None):
    if not fields:fields=list(dict.fromkeys(k for row in rows for k in row)) or ['status']
    with Path(path).open('w',newline='',encoding='utf-8') as target:
        writer=csv.DictWriter(target,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)


def _image(fig):
    import base64
    import matplotlib.pyplot as plt
    buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=120,bbox_inches='tight');plt.close(fig)
    return 'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()


def report(root,meta,arrays,comparison=None,refs=None):
    """Keep the real-data report separate from synthetic test results."""
    import html
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from analysis import MARKER_COLORS
    esc=lambda value:html.escape(str(value))
    names=dict(declared_rate_hz='Sampling rate (Hz)',nonfinite_values='Invalid values',
        timestamp_reversals='Times moving backward',duplicate_timestamps='Repeated times',
        largest_sample_gap_ms='Largest sample interval (ms)',sample_to_read_p99_ms='Delay until read p99 (ms)',
        last_sample_age_ms='Last EEG sample age at stop (ms)',
        matched='Matched samples',different_values='Different values',nonfinite_pairs='Pairs with invalid values',
        max_difference='Largest value difference',largest_timestamp_difference_ms='Largest time difference (ms)')
    def table(rows):
        if not rows:return '<p>No measurements available.</p>'
        fields=list(rows[0])
        return '<div class="scroll"><table><tr>'+''.join('<th>'+esc(names.get(k,k.replace('_',' ')))+'</th>' for k in fields)+'</tr>'+''.join('<tr>'+''.join('<td>'+('—' if r[k] is None else esc(r[k]))+'</td>' for k in fields)+'</tr>' for r in rows)+'</table></div>'
    root=Path(root);start=meta['start'];eeg=arrays['eeg_values'];stamps=arrays['eeg_corrected']
    parts=['<!doctype html><html lang="en"><meta charset="utf-8"><title>Real EEG — Python LSL capture</title>',
        '<style>body{font:16px system-ui;max-width:1250px;margin:30px auto;padding:20px;background:#f6f8fc;color:#17233a}section{background:white;padding:24px;margin:18px 0}img{max-width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}table{border-collapse:collapse;font-size:13px}.scroll{overflow:auto}p{line-height:1.5}</style>',
        '<h1>Real EEG — Python LSL capture</h1>',
        '<p>Stream: '+esc(meta['streams']['eeg']['name'])+'. '+esc(meta['status'])+'. '+esc(meta['error'])+'</p>']
    capture_rows=[]
    for role,info in meta['streams'].items():
        raw=arrays[role+'_raw'];delay=(arrays[role+'_arrival']-arrays[role+'_corrected'])*1000
        capture_rows.append(dict(stream=role,name=info['name'],channels=info['channels'],declared_rate_hz=info['rate_hz'],
            samples=len(raw),nonfinite_values=int(np.count_nonzero(~np.isfinite(arrays[role+'_values']))) if role=='eeg' else None,
            timestamp_reversals=int(np.sum(np.diff(raw)<0)),duplicate_timestamps=int(np.sum(np.diff(raw)==0)),
            largest_sample_gap_ms=round(float(np.max(np.diff(raw))*1000),4) if len(raw)>1 else None,
            last_sample_age_ms=round(float((meta['end']-arrays[role+'_corrected'][-1])*1000),4) if role=='eeg' and len(raw) else None,
            sample_to_read_p99_ms=round(float(np.percentile(delay,99)),4) if len(delay) else None))
    parts.append('<section><h2>Recorded EEG and markers</h2>')
    if len(eeg):
        fig,axes=plt.subplots(2,1,figsize=(13,7),constrained_layout=True)
        elapsed=stamps-start;window=(elapsed>=0)&(elapsed<=min(10,meta['end']-start))
        label=(meta['streams']['eeg'].get('labels') or ['Channel 1'])[0]
        axes[0].plot(elapsed[window],eeg[window,0],lw=.8,label='Received EEG — '+label)
        axes[0].set(title='Recorded EEG — first channel, first 10 seconds',ylabel='Original stream value',xlabel='Time since capture start (s)')
        if 'marker' in meta['streams']:
            values=arrays['marker_values'];times=arrays['marker_corrected']-start
            keys=[' / '.join(row) for row in values];labels=list(dict.fromkeys(keys))
            for key,t in zip(keys,times):
                if 0<=t<=10:
                    index=('A','B','C').index(key) if key in ('A','B','C') else 3+labels.index(key)
                    color=MARKER_COLORS[index%len(MARKER_COLORS)]
                    axes[0].axvline(t,color=color,lw=2.5,ls='--',alpha=.8)
                    axes[0].annotate(key,xy=(t,.98),xycoords=('data','axes fraction'),xytext=(3,-3),textcoords='offset points',
                        va='top',color=color,fontsize=10,weight='bold',bbox=dict(facecolor='white',edgecolor='none',alpha=.8))
        axes[0].legend(fontsize=8)
        axes[1].plot(elapsed,(arrays['eeg_arrival']-stamps)*1000,lw=.6)
        axes[1].set(title='Delay until Python read each EEG sample',ylabel='Delay (ms)',xlabel='Time since capture start (s)')
        for ax in axes:ax.grid(alpha=.2)
        parts.append('<img src="'+_image(fig)+'" alt="Actual EEG values, optional received markers and measured read delays">')
    else:parts.append('<p>No EEG samples were received.</p>')
    parts.append(table(capture_rows)+'<p>p99 means 99% of samples had a delay at or below this value. Marker status: '+esc(meta['marker_status'])+'. No artificial EEG changes or markers are added. Marker positions use their own LSL clock correction.</p></section>')
    metrics=[dict(section='capture',**row) for row in capture_rows]
    marker_rows=[]
    if 'marker' in meta['streams']:
        for i,raw in enumerate(arrays['marker_raw']):
            marker_rows.append(dict(event=i,value=' / '.join(arrays['marker_values'][i]),source_timestamp=float(raw),
                sample_time_s=float(arrays['marker_corrected'][i]-start),read_time_s=float(arrays['marker_arrival'][i]-start),
                reference_timestamp=None,reference_value=None,reference_time_difference_ms=None,values_match=None))
    if comparison:
        paired=comparison['paired'];a,b=paired['captured'],paired['reference']
        summary=dict(matched=len(a),python_samples_in_shared_range=paired['captured_in_range'],
            reference_samples_in_shared_range=paired['reference_in_range'],python_without_reference_match=paired['unmatched_captured'],
            reference_without_python_match=paired['unmatched_reference'],
            largest_timestamp_difference_ms=float(np.max(np.abs(comparison['timestamp_difference_ms']))))
        parts.append('<section><h2>Python and reference — same EEG samples</h2>')
        fig,axes=plt.subplots(2,1,figsize=(13,7),constrained_layout=True)
        origin=arrays['eeg_raw'][a[0]];view=arrays['eeg_raw'][a]-origin<=5
        axes[0].plot(arrays['eeg_raw'][a[view]]-origin,eeg[a[view],0],lw=1.3,label='Python')
        axes[0].plot(refs['eeg']['raw'][b[view]]-origin,refs['eeg']['values'][b[view],0],lw=1,ls='--',label='Reference')
        axes[0].set(title='Same channel — original sample times, no curve shifting',ylabel='Original value',xlabel='Time from shared source origin (s)');axes[0].legend()
        errors=np.array([r['max_difference'] if r['max_difference'] is not None else np.nan for r in comparison['rows']])
        finite=errors[np.isfinite(errors)]
        image=axes[1].imshow(np.ma.masked_invalid(errors[None,:]),aspect='auto',vmin=0,
            vmax=float(np.max(finite)) or 1 if len(finite) else 1,cmap=plt.get_cmap('viridis').with_extremes(bad='#ddd'))
        axes[1].set(title='Largest value difference per channel — zero means values match',xlabel='Channel',yticks=[])
        axes[1].set_xticks(range(len(errors)),[str(i+1) for i in range(len(errors))],rotation=90,fontsize=6)
        fig.colorbar(image,ax=axes[1],shrink=.8)
        parts.append('<img src="'+_image(fig)+'" alt="Python and reference overlay and per-channel value differences">'+table([summary]))
        parts.append('<p>'+esc(comparison['description'])+'. Matching uses original source timestamps within 1 microsecond, only in the shared recording time range. Samples outside this range are not counted as missing. Gray cells mean no valid values to compare. No resampling, value scaling or delay fitting is applied.</p>')
        parts.append('<p>All channel labels/order verified: '+str(comparison['channel_labels_verified'])+'. Known channel labels checked: '+str(comparison['known_labels_checked'])+'. Where labels are unavailable, the comparison uses the original channel order.</p>')
        parts.append('<details><summary>Every channel</summary>'+table(comparison['rows'])+'</details>')
        metrics.extend(dict(section='reference_channel',**r) for r in comparison['rows'])
        metrics.append(dict(section='reference_summary',**summary))
        parts.append('<h3>Markers in Python and reference</h3>')
        if comparison.get('markers'):
            pair=comparison['markers'];ma,mb=pair['captured'],pair['reference']
            marker_summary=dict(matched=len(ma),python_without_reference_match=pair['unmatched_captured'],
                reference_without_python_match=pair['unmatched_reference'],different_marker_values=pair['different_values'],
                largest_timestamp_difference_ms=float(np.max(np.abs(pair['timestamp_difference_ms']))) if len(ma) else None)
            parts.append(table([marker_summary]));metrics.append(dict(section='reference_markers',**marker_summary))
            for i,j in zip(ma,mb):
                marker_rows[int(i)]['reference_timestamp']=float(refs['marker']['raw'][j])
                marker_rows[int(i)]['reference_value']=' / '.join(refs['marker']['values'][j])
                marker_rows[int(i)]['reference_time_difference_ms']=float((arrays['marker_raw'][i]-refs['marker']['raw'][j])*1000)
                marker_rows[int(i)]['values_match']=bool(np.array_equal(arrays['marker_values'][i],refs['marker']['values'][j]))
        else:parts.append('<p>'+esc(comparison.get('marker_error','No marker comparison available.'))+'</p>')
        parts.append('</section>')
    else:parts.append('<section><h2>Reference comparison</h2><p>No reference supplied yet. Record the same EEG and marker streams simultaneously in LabRecorder, then run --compare-eeg with --reference recording.xdf.</p></section>')
    parts.append('<section><h2>After viewing the results</h2><p>This records the existing device stream, preserving original values and timestamps. There is no sample counter added by the amplifier here, so timestamp gaps alone do not prove sample loss. A second recording checks the shared LSL data path; physical stimulus timing requires a separately recorded trigger or sensor.</p><p>Files: recordings.npz, metrics.csv, markers.csv and this summary.html. Raw EEG stays local.</p></section></html>')
    _csv(root/'metrics.csv',metrics);_csv(root/'markers.csv',marker_rows,
        ['event','value','source_timestamp','sample_time_s','read_time_s','reference_timestamp','reference_value','reference_time_difference_ms','values_match'])
    (root/'summary.html').write_text('\n'.join(parts),encoding='utf-8')
