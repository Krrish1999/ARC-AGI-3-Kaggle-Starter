"""Bounded repeated-rectangle lattice evidence and exact pixel crop extraction.
No game IDs, palette roles, terrain rules, paths, fixed origins or tile sizes.
Pitch-equals-component-size is a hypothesis, not a universal board detector.
Dependency-free so this module can be injected into the isolated Python tool.
"""


def _gc_grid(grid):
    if type(grid) not in (list, tuple) or not grid or len(grid) > 256:
        return None
    if type(grid[0]) not in (list, tuple) or not grid[0] or len(grid[0]) > 256:
        return None
    w = len(grid[0])
    if len(grid)*w > 8192 or any(type(row) not in (list, tuple) or len(row) != w or any(type(v) is not int or not 0 <= v < 16 for v in row) for row in grid):
        return None
    return grid


def _gc_components(grid):
    h,w = len(grid),len(grid[0])
    seen = [[False]*w for _ in range(h)]
    nodes = []
    for r in range(h):
        for c in range(w):
            if seen[r][c]:
                continue
            if len(nodes) >= 1024:
                return None
            queue = [(r,c)];seen[r][c] = True
            r0=r1=r;c0=c1=c
            for rr,cc in queue:
                r0=min(r0,rr);r1=max(r1,rr);c0=min(c0,cc);c1=max(c1,cc)
                for nr,nc in ((rr-1,cc),(rr+1,cc),(rr,cc-1),(rr,cc+1)):
                    if 0<=nr<h and 0<=nc<w and not seen[nr][nc] and grid[nr][nc]==grid[r][c]:
                        seen[nr][nc]=True;queue.append((nr,nc))
            nodes.append({'id':len(nodes),'bbox_rc':[r0,c0,r1,c1],'pixels':len(queue)})
    return nodes


def grid_coverage_candidates(grid):
    """Find repeated component-box candidates. Extents are inclusive.

At least four solid rectangles, >=2 distinct row AND column starts, and a
common size/residue are required. Non-solid same-sized aligned components can
extend the supported extent, but do not supply the initial repetition evidence.
Singleton markers never define a pitch. Thin HUD lines are not assigned a role.
Multiple aligned regions may share a candidate: grouping remains unresolved.
"""
    result = {'status':'ok','candidates':[],'candidates_truncated':False,
              'limits':{'cells':8192,'dimension':256,'components':1024,'candidates':8},
              'coverage':'observed repeated-component evidence only, not proof of complete board bounds'}
    g = _gc_grid(grid)
    if g is None:
        result.update(status='unavailable',reason='invalid_or_oversized_grid');return result
    nodes = _gc_components(g)
    if nodes is None:
        result.update(status='unavailable',reason='component_limit');return result
    groups = {}
    for n in nodes:
        r0,c0,r1,c1=n['bbox_rc'];ph,pw=r1-r0+1,c1-c0+1
        if ph<2 or pw<2:
            continue
        key=(ph,pw,r0%ph,c0%pw)
        groups.setdefault(key,[]).append(n)
    candidates = []
    for (ph,pw,rr,cc),group in groups.items():
        solid=[n for n in group if n['pixels']==ph*pw]
        if len(solid)<4 or len({n['bbox_rc'][0] for n in solid})<2 or len({n['bbox_rc'][1] for n in solid})<2:
            continue
        r0=min(n['bbox_rc'][0] for n in group);c0=min(n['bbox_rc'][1] for n in group)
        r1=max(n['bbox_rc'][2] for n in group);c1=max(n['bbox_rc'][3] for n in group)
        shape=[(r1-r0+1)//ph,(c1-c0+1)//pw]
        partial=[]
        # Nearby smaller components at an aligned start can indicate clipped or
        # partial edge cells. Report, don't extend a rectangle through them.
        for n in nodes:
            a,b,c,d=n['bbox_rc'];nh,nw=c-a+1,d-b+1
            outside=a<r0 or b<c0 or c>r1 or d>c1
            near=r0-ph<=a<=r1+ph and c0-pw<=b<=c1+pw
            if outside and near and nh<=ph and nw<=pw and (nh<ph or nw<pw) and a%ph==rr and b%pw==cc:
                partial.append(n['id'])
        candidates.append({'origin_rc':[r0,c0],'pitch_rc':[ph,pw],'shape_rc':shape,
            'bbox_rc':[r0,c0,r1,c1],'solid_support':len(solid),'box_support':len(group),
            'support_component_ids':[n['id'] for n in group],
            'nearby_partial_component_ids':partial[:8],'nearby_partial_count':len(partial),
            'interpretation':'candidate only; pitch equals component size; separate aligned regions may be grouped; roles/identity unresolved'})
    candidates.sort(key=lambda x:(-x['solid_support'],-x['box_support'],x['pitch_rc'],x['origin_rc']))
    result['candidates_truncated']=len(candidates)>8
    result['candidates']=candidates[:8]
    return result


def _gc_pair(value, name, positive=False):
    if type(value) not in (list,tuple) or len(value)!=2 or any(type(x) is not int for x in value):
        raise TypeError(name+' must be an integer row/column pair; no rounding')
    if any(x < (1 if positive else 0) for x in value):
        raise ValueError(name+' is outside the supported range')
    return tuple(value)


def extract_grid_view(grid, colors, *, candidate=None, origin_rc=None, pitch_rc=None, shape_rc=None, topology=None):
    """Return requested raw tiles/samples AND a geometric coverage audit.

Never silently expand/clip the requested crop, classify walkability, infer a
hazard model, or certify an unreachable result. Explicit candidate selection is
required if multiple interpretations exist. Manual geometry works without a
candidate, but its coverage is then unverified. Geometry is re-derived from raw
pixels, not from the caller's mutable cached candidate dictionary.
"""
    g=_gc_grid(grid)
    if g is None:
        raise ValueError('grid view requires a valid grid within the documented bounds')
    report=grid_coverage_candidates(g)
    choices=report['candidates']
    manual=any(x is not None for x in (origin_rc,pitch_rc,shape_rc))
    if candidate is not None and manual:
        raise ValueError('choose a candidate OR supply all three geometry pairs')
    if not manual:
        if report['status']!='ok' or not choices:
            raise ValueError('no supported candidate; supply explicit geometry and treat coverage as unverified')
        if candidate is None:
            if len(choices)!=1 or report['candidates_truncated']:
                raise ValueError('ambiguous candidates; select candidate=index or supply explicit geometry')
            candidate=0
        if type(candidate) is not int or not 0<=candidate<len(choices):
            raise ValueError('candidate must be an in-range integer index')
        chosen=choices[candidate]
        origin_rc,pitch_rc,shape_rc=(chosen[k] for k in ('origin_rc','pitch_rc','shape_rc'))
    elif any(x is None for x in (origin_rc,pitch_rc,shape_rc)):
        raise ValueError('explicit geometry requires origin_rc, pitch_rc AND shape_rc')
    r0,c0=_gc_pair(origin_rc,'origin_rc');ph,pw=_gc_pair(pitch_rc,'pitch_rc',True);nr,nc=_gc_pair(shape_rc,'shape_rc',True)
    r1,c1=r0+nr*ph-1,c0+nc*pw-1
    if r1>=len(g) or c1>=len(g[0]):
        raise ValueError('requested full tiles extend outside the frame; no silent clipping')
    matching=[x for x in choices if x['pitch_rc']==[ph,pw] and (r0-x['origin_rc'][0])%ph==0 and (c0-x['origin_rc'][1])%pw==0]
    omissions=[]
    for x in matching:
        a,b,c,d=x['bbox_rc']
        if a<r0 or b<c0 or c>r1 or d>c1:
            omissions.append({'observed_bbox_rc':x['bbox_rc'],'suggested_origin_rc':x['origin_rc'],
                              'suggested_shape_rc':x['shape_rc'],
                              'outside_sides':[name for name,test in (('top',a<r0),('left',b<c0),('bottom',c>r1),('right',d>c1)) if test]})
    coverage='incomplete_observed_extent' if omissions else ('covers_observed_extent' if matching else 'unverified')
    tiles=[];samples=[];mixed=[];color_cells={};color_counts=[]
    # Even-sized tiles have no central pixel. Explicit convention: upper/left
    # of the central pair. Preserve the full tile, never synthesize a color.
    sr,sc=(ph-1)//2,(pw-1)//2
    for r in range(nr):
        row=[];srow=[];counts_row=[]
        for c in range(nc):
            a,b=r0+r*ph,c0+c*pw
            tile=[''.join(colors[v] for v in line[b:b+pw]) for line in g[a:a+ph]]
            row.append(tile);srow.append(colors[g[a+sr][b+sc]])
            counts={}
            for line in tile:
                for color in line:counts[color]=counts.get(color,0)+1
            for color in sorted(counts):color_cells.setdefault(color,[]).append([r,c])
            counts_row.append(counts)
            if len(counts)>1:mixed.append([r,c])
        tiles.append(row);samples.append(srow);color_counts.append(counts_row)
    view = {'origin_rc':[r0,c0],'pitch_rc':[ph,pw],'shape_rc':[nr,nc],
            'bbox_rc':[r0,c0,r1,c1],'sample_offset_rc':[sr,sc],
            'samples':samples,'tiles':tiles,'mixed_cells_rc':mixed,
            'sample_rows':[''.join(row) for row in samples],
            'color_cells':color_cells,'color_counts':color_counts,
            'schema':{'samples':'[r][c] single observed color',
                      'sample_rows':'[r] string of sample colors',
                      'tiles':'[r][c][pixel_row] string of raw pixels; NOT a color list',
                      'color_cells':'color -> distinct [r,c] tiles containing that pixel color',
                      'color_counts':'[r][c] -> color: pixel count'},
            'coverage':coverage,'omitted_extent_evidence':omissions,
            'candidate_report_status':report['status'],'candidate_report_truncated':report['candidates_truncated'],
            'nearby_partial_count':sum(x['nearby_partial_count'] for x in matching),
            'meaning':'samples are observed pixel colors, not semantic terrain; coverage is not proof of complete board or safe reachability'}

    view['topology'] = None if topology is None else _gc_topology(view,topology,colors)
    return view


def grid_coverage_note(report):
    if report['status']!='ok' or not report['candidates']:
        return ''
    parts=[]
    for i,x in enumerate(report['candidates'][:2]):
        parts.append(str(i)+': origin='+str(x['origin_rc'])+' pitch='+str(x['pitch_rc'])+' rows,cols='+str(x['shape_rc'])+' inclusive_bbox='+str(x['bbox_rc']))
    return ('[grid_coverage] '+'; '.join(parts)+
        '; candidate extents, not proven board bounds. current_frame.grid_view() audits crop coverage; include observed last rows/columns before concluding unreachable.')[:650]


def grid_view_warning(view):
    if view['coverage']=='incomplete_observed_extent':
        return '[grid_coverage] requested crop omits aligned observed extent: '+str(view['omitted_extent_evidence'][:2])[:420]+'; do not conclude unreachable from this crop alone. No crop expansion or action performed.'
    if view['coverage']=='unverified':
        return '[grid_coverage] no matching repeated-component extent; requested samples returned, coverage UNVERIFIED. No completeness/safety claim.'
    if view['nearby_partial_count'] or view['candidate_report_truncated']:
        return '[grid_coverage] covers selected observed extent, but nearby partial components or truncated candidates leave bounds unresolved.'
    return ''


def _gc_topology(view, spec, colors):
    """Static graph under EXPLICIT caller color/stride hypotheses, never safety.

Node samples select candidate floor; pixel presence selects occupied nodes.
An occupied node's terrain remains unknown. Intermediate connector samples
must match the caller's connector colors. No diagonals, wraps or inferred roles.
"""
    keys={'node_colors','occupied_colors','connector_colors','stride_rc','start_rc','goal_rc'}
    if type(spec) is not dict or any(type(k) is not str for k in spec) or set(spec)!=keys:
        raise ValueError('topology requires exactly node_colors, occupied_colors, connector_colors, stride_rc, start_rc, goal_rc')
    def palette(name):
        xs=spec[name]
        if type(xs) not in (list,tuple,str) or len(xs)>16:
            raise TypeError(name+' must be at most 16 single-character colors')
        if any(type(x) is not str or len(x)!=1 or x not in colors for x in xs):
            raise ValueError(name+' contains an invalid color')
        return set(xs)
    floor=palette('node_colors');occupied=palette('occupied_colors');connector=palette('connector_colors')
    stride=_gc_pair(spec['stride_rc'],'stride_rc',True)
    if max(stride)>256:raise ValueError('stride exceeds supported dimension')
    start=_gc_pair(spec['start_rc'],'start_rc');goal=_gc_pair(spec['goal_rc'],'goal_rc')
    nr,nc=view['shape_rc']
    if any(r>=nr or c>=nc for r,c in (start,goal)):raise ValueError('endpoint outside requested view')
    occupied_rc={tuple(rc) for color in occupied for rc in view['color_cells'].get(color,[])}
    nodes=sorted({(r,c) for r,row in enumerate(view['samples']) for c,color in enumerate(row) if color in floor}|occupied_rc)
    index={rc:i for i,rc in enumerate(nodes)};neighbors=[[] for rc in nodes]
    for i,(r,c) in enumerate(nodes):
        for dr,dc in ((-stride[0],0),(stride[0],0),(0,-stride[1]),(0,stride[1])):
            dest=(r+dr,c+dc)
            if dest not in index:continue
            steps=abs(dr)+abs(dc);rr=(dr>0)-(dr<0);cc=(dc>0)-(dc<0)
            if all(view['samples'][r+k*rr][c+k*cc] in connector for k in range(1,steps)):
                neighbors[i].append(index[dest])
    missing=[name for name,rc in (('start',start),('goal',goal)) if rc not in index]
    connected=None
    if not missing:
        seen={index[start]};queue=[index[start]]
        for i in queue:
            for j in neighbors[i]:
                if j not in seen:seen.add(j);queue.append(j)
        connected=index[goal] in seen
    return {'status':'missing_endpoint' if missing else ('connected' if connected else 'disconnected'),
            'nodes_rc':[list(rc) for rc in nodes],'neighbors':neighbors,
            'start_index':index.get(start),'goal_index':index.get(goal),
            'missing_endpoints':missing,'connected':connected,
            'occupied_rc':[list(rc) for rc in sorted(occupied_rc)],
            'underlay_unknown_rc':[list(rc) for rc in sorted(occupied_rc)],
            'hypothesis':{'node_colors':sorted(floor),'occupied_colors':sorted(occupied),
                          'connector_colors':sorted(connector),'stride_rc':list(stride)},
            'meaning':'caller-hypothesis static topology ONLY; occupied nodes retained, underlying terrain unknown; no hazard, collision, resource, route-safety or complete-board certification'}


def grid_contract_note(view):
    note='[grid_contract] sample_rows[r] is a string; samples[r][c] is one pixel color. color_cells[color] lists tiles with that color; color_counts[r][c] counts pixels. tiles[r][c] is a LIST OF ROW STRINGS, not colors.'
    t=view['topology']
    if t is not None:
        note+=' Topology='+t['status']+'; nodes='+str(len(t['nodes_rc']))+'; missing='+str(t['missing_endpoints'])+'.'
    return note+' Occupancy is not terrain; static connectivity is NOT route safety.'
