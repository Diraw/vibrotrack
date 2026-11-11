"""Dense disc extraction primitives (v1 detector).

Inputs are never modified.  This module is intentionally free of lab-specific
paths and case tables; the CLI/pipeline supplies a mask path and scale levels.
"""
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

cv2.setNumThreads(1)

# Common Basler 16-bit range used by the original apparatus. Override through
# the apparatus config for other cameras.
DEFAULT_SCALE_LEVELS = (128.0, 28224.0)

def dump(p,obj):
    p=Path(p); q=p.with_suffix(p.suffix+'.tmp'); q.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf8'); q.replace(p)

PRODUCTION_LEVELS = DEFAULT_SCALE_LEVELS
_U8_LUTS={}

def scale_to_u8(a,levels):
    """uint16 -> uint8 with the historical truncating mapping.

    Reference:
        clip((a.astype(float32) - lo) * 255/(hi-lo), 0, 255).astype(uint8)
    The production Basler levels (128, 28224) use cv2.subtract + convertScaleAbs:
    subtract saturates below lo at 0, convertScaleAbs scales/saturates at 255,
    and beta=-0.5 reproduces the old truncation bit-for-bit.  Other integer
    levels use an exact 65536-entry LUT; non-integer levels fall back to the
    reference expression.
    """
    lo,hi=float(levels[0]),float(levels[1])
    if hi<=lo or not (lo.is_integer() and hi.is_integer()):
        return np.clip((a.astype(np.float32)-lo)*(255/(hi-lo)),0,255).astype(np.uint8)
    if (lo,hi)==PRODUCTION_LEVELS:
        d=cv2.subtract(a,lo)
        return cv2.convertScaleAbs(d,alpha=255.0/(hi-lo),beta=-0.5)
    lut=_U8_LUTS.get((lo,hi))
    if lut is None:
        lut=np.clip((np.arange(65536,dtype=np.float32)-lo)*(255/(hi-lo)),0,255).astype(np.uint8)
        _U8_LUTS[(lo,hi)]=lut
    return np.take(lut,a)

def read_image(p,levels,strict=False):
    # 2026-11-06: these Basler files are uncompressed 16-bit, 2400 strips of one row.
    # cv2.imdecode costs 54 ms/frame on them; the strip reader costs 3-4 ms and is
    # byte-identical (verified 15/15 files). Fallback keeps the old path.
    try:
        from .fast_tiff import read as _fastread
    except ImportError:                 # flat execution, e.g. a one-off script
        from fast_tiff import read as _fastread
    try:
        a = _fastread(p)
    except Exception as exc:
        if Path(p).stat().st_size == 0:
            raise IOError('0-byte frame in the middle of the sequence: ' + str(p)) from exc
        a = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_UNCHANGED)
    if a is None: raise OSError('Cannot decode '+str(p))
    if a.dtype == np.uint8:
        return a
    if strict:
        lo,hi=levels
        return np.clip((a.astype(np.float32)-lo)*(255/(hi-lo)),0,255).astype(np.uint8)
    return scale_to_u8(a,levels)

def mask_bbox(mask,margin=48):
    """Full-resolution bbox of the valid region, padded for the 47px template."""
    ys,xs=np.where(mask>0)
    if not len(xs): return None
    return (max(0,int(xs.min())-margin),min(mask.shape[1],int(xs.max())+margin+1),
            max(0,int(ys.min())-margin),min(mask.shape[0],int(ys.max())+margin+1))

def template(radius=18.):
    y,x=np.mgrid[-23:24,-23:24]; r=np.hypot(x,y)
    t=np.clip((radius+.7-r)/1.4,0,1).astype(np.float32)
    t[r<.38*radius]=.72
    return t

TEMPLATE=template()
ANGLES=np.linspace(0,2*np.pi,64,endpoint=False).astype(np.float32)
CS=np.cos(ANGLES); SN=np.sin(ANGLES)
RAD=np.arange(28,45,.75,dtype=np.float32)
PY,PX=np.mgrid[-23:24,-23:24].astype(np.float32)
INNER=(PX*PX+PY*PY)<22**2

def detect(im,mask,dist=None,bbox=None):
    small=cv2.resize(im,None,fx=.5,fy=.5,interpolation=cv2.INTER_AREA)
    # Match only where the mask can produce a valid centre.  The 24 px
    # half-res margin keeps the 47x47 template fully inside the crop.
    if bbox is not None:
        x0,x1,y0,y1=bbox
        cx0=max(0,x0//2-24); cx1=min(small.shape[1],x1//2+24)
        cy0=max(0,y0//2-24); cy1=min(small.shape[0],y1//2+24)
        sub=small[cy0:cy1,cx0:cx1]
    else:
        cx0=cy0=0; sub=small
    # Keep the historical /255: dropping it (mathematically a no-op for
    # TM_CCOEFF_NORMED) changes confidence by ~1.5e-6 and breaks bit-exact
    # comparison with the archived chunks.
    score=cv2.matchTemplate(sub.astype(np.float32)/255,TEMPLATE,cv2.TM_CCOEFF_NORMED)
    peaks=(score==cv2.dilate(score,np.ones((27,27),np.uint8)))&(score>.27)
    yy,xx=np.where(peaks); conf=score[yy,xx]
    xy=np.column_stack(((xx+cx0+23)*2,(yy+cy0+23)*2)).astype(np.float32)
    keep=mask[np.clip(xy[:,1].astype(int),0,mask.shape[0]-1),np.clip(xy[:,0].astype(int),0,mask.shape[1]-1)]>0
    xy=xy[keep]; conf=conf[keep]
    if not len(xy): return empty_detection()
    # Radial negative-gradient edge sampling, all particles in one remap.
    mx=xy[:,0,None,None]+CS[None,:,None]*RAD[None,None,:]
    my=xy[:,1,None,None]+SN[None,:,None]*RAD[None,None,:]
    prof=cv2.remap(im,mx.reshape(len(xy),-1),my.reshape(len(xy),-1),cv2.INTER_LINEAR).reshape(len(xy),64,-1).astype(np.float32)
    grad=prof[:,:,:-2]-prof[:,:,2:]
    k=grad.argmax(2); strength=np.take_along_axis(grad,k[:,:,None],2)[:,:,0]
    rr=RAD[1:-1][k]
    A=np.column_stack((CS,SN,np.ones(64)))
    valid=strength>20
    coef=np.zeros((len(xy),3)); rmse=np.full(len(xy),99.)
    for _ in range(3):
        w=valid.astype(float)
        lhs=np.einsum('ni,ij,ik->njk',w,A,A)+np.eye(3)[None]*1e-5
        rhs=np.einsum('ni,ni,ij->nj',w,rr,A)
        coef=np.linalg.solve(lhs,rhs[...,None])[...,0]
        resid=rr-coef@A.T
        valid=(strength>20)&(np.abs(resid)<2.7)
    ns=valid.sum(1); rmse=np.sqrt((resid**2*valid).sum(1)/np.maximum(ns,1))
    shift=np.linalg.norm(coef[:,:2],axis=1)
    center_valid=(ns>=30)&(rmse<1.8)&(shift<5)&(coef[:,2]>30)&(coef[:,2]<42)
    xy+=coef[:,:2].astype(np.float32)
    radius=coef[:,2].astype(np.float32)
    # Reject invalid centers, but report count separately via caller pilot diagnostics.
    keep=center_valid & (mask[np.clip(xy[:,1].astype(int),0,mask.shape[0]-1),np.clip(xy[:,0].astype(int),0,mask.shape[1]-1)]>0)
    xy,conf,radius,rmse,ns=xy[keep],conf[keep],radius[keep],rmse[keep],ns[keep]
    n=len(xy)
    if not n: return empty_detection()
    # Bright disc interior excludes boundary and touching neighbors.
    mx=xy[:,0,None,None]+PX; my=xy[:,1,None,None]+PY
    patch=cv2.remap(im,mx.reshape(n,-1),my.reshape(n,-1),cv2.INTER_LINEAR).reshape(n,47,47).astype(np.float32)
    bright=np.percentile(patch[:,INNER],80,axis=1)
    dark=(patch<bright[:,None,None]*.54)&INNER[None]
    weight=np.maximum(bright[:,None,None]*.54-patch,0)*dark
    area=dark.sum((1,2)); mass=weight.sum((1,2))
    dx=(weight*PX).sum((1,2))/np.maximum(mass,1); dy=(weight*PY).sum((1,2))/np.maximum(mass,1)
    offset=np.hypot(dx,dy)
    angle_valid=(area>=100)&(area<=380)&(offset>=2)&(offset<=16)&(bright>85)
    theta=np.arctan2(-dy,dx).astype(np.float32); theta[~angle_valid]=np.nan
    nearest=cKDTree(xy).query(xy,k=2)[0][:,1] if n>1 else np.array([np.inf])
    bd=dist[np.clip(xy[:,1].astype(int),0,mask.shape[0]-1),np.clip(xy[:,0].astype(int),0,mask.shape[1]-1)] if dist is not None else np.full(n,np.nan)
    bits=(~angle_valid).astype(np.uint16)|((nearest<1.85*radius).astype(np.uint16)<<1)|((bd<radius+4).astype(np.uint16)<<2)
    return dict(xy_px=xy,radius_px=radius,theta_rad=theta,center_valid=np.ones(n,bool),angle_valid=angle_valid,confidence=conf.astype(np.float32),edge_rmse_px=rmse.astype(np.float32),marker_offset_px=offset.astype(np.float32),marker_area_px=area.astype(np.int16),neighbor_distance_px=nearest.astype(np.float32),boundary_distance_px=bd.astype(np.float32),quality_bits=bits)

def empty_detection():
    z=np.zeros((1,1),np.uint8)
    return {k:np.empty((0,2) if k=='xy_px' else 0,dtype=np.float32) for k in ['xy_px','radius_px','theta_rad','center_valid','angle_valid','confidence','edge_rmse_px','marker_offset_px','marker_area_px','neighbor_distance_px','boundary_distance_px','quality_bits']}

def _two_nearest(tree,points):
    k=2 if len(tree.data)>=2 else 1
    d,i=tree.query(points,k=k)
    if k==1:
        d=np.column_stack([d,np.full(len(d),np.inf)])
        i=np.column_stack([i,np.full(len(i),-1)])
    return d,i

def link(prev,cur,nextid,prevoffset):
    n=len(cur['xy_px']); tids=np.arange(nextid,nextid+n,dtype=np.int32); nextid+=n
    pd=np.full(n,-1,np.int64); margin=np.full(n,np.nan,np.float32); step=np.full(n,np.nan,np.float32); dt=np.full(n,np.nan,np.float32)
    if prev is not None and len(prev['xy_px']) and n:
        a=prev['xy_px']; b=cur['xy_px']
        gate=float(min(np.median(prev['radius_px']),np.median(cur['radius_px']))*.9)
        # The gate graph is sparse (about one candidate per particle at 10 fps).
        # The old accept condition is exactly "mutual nearest + gap", so the
        # dense NxN Hungarian is unnecessary.  Query both directions and keep
        # the mutual pairs; the resulting segment decisions are identical.
        da,ia=_two_nearest(cKDTree(b),a)
        db,ib=_two_nearest(cKDTree(a),b)
        for c in range(n):
            r=int(ib[c,0]); dd=float(db[c,0])
            if r>=0 and np.isfinite(dd):
                margin[c]=min(float(da[r,1]),float(db[c,1]))-dd
        for r in range(len(a)):
            c=int(ia[r,0])
            if c<0: continue
            dd=float(da[r,0])
            if dd>=gate or int(ib[c,0])!=r: continue
            gap=min(float(da[r,1]),float(db[c,1]))-dd
            if gap>max(5.,.22*gate):
                # Use the float32 coordinates, not the float64 KDTree distance,
                # so step_px keeps the same bits as the old implementation.
                tids[c]=prev['segment_id'][r]; pd[c]=prevoffset+r
                step[c]=float(np.linalg.norm(a[r]-b[c]))
                if prev['angle_valid'][r] and cur['angle_valid'][c]: dt[c]=np.arctan2(np.sin(cur['theta_rad'][c]-prev['theta_rad'][r]),np.cos(cur['theta_rad'][c]-prev['theta_rad'][r]))
    cur.update(segment_id=tids,previous_detection=pd,link_valid=pd>=0,link_margin_px=margin,step_px=step,dtheta_rad=dt)
    return nextid

def overlay(im,d,path,label):
    out=cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
    for i,(x,y) in enumerate(d['xy_px']):
        color=(0,220,0) if d['angle_valid'][i] else (0,120,255)
        cv2.circle(out,(round(float(x)),round(float(y))),round(float(d['radius_px'][i])),color,1)
        cv2.circle(out,(round(float(x)),round(float(y))),2,(0,0,255),-1)
        if d['angle_valid'][i]:
            t=d['theta_rad'][i]; q=(round(float(x+24*np.cos(t))),round(float(y-24*np.sin(t))))
            cv2.line(out,(round(float(x)),round(float(y))),q,color,2)
    cv2.putText(out,label,(25,45),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    Image.fromarray(cv2.cvtColor(out,cv2.COLOR_BGR2RGB)).save(path,quality=90)

def _numeric_key(p):
    m=re.search(r'(\d+)(?!.*\d)',Path(p).stem)
    return (int(m.group(1)) if m else -1, str(p))

def files_for(case, strict=False):
    # 2026-11-06: sorting filenames lexicographically silently reindexes frames when the
    # numbering is not zero-padded, or while files are being moved. Sort by the numeric tail.
    exts = tuple(x.lower() for x in (case.get('extensions') or ('.tiff', '.tif')))
    root = Path(case['path'])
    # os.scandir returns the directory entry attributes with the listing, so
    # is_file()/suffix do not cost a stat per frame (3.4 ms x 9000 on the USB
    # archive = 31 s). Path.is_file() would stat every entry again.
    with os.scandir(root) as entries:
        files = [Path(e.path) for e in entries
                 if e.is_file() and Path(e.name).suffix.lower() in exts]
    files.sort(key=_numeric_key)
    # 2026-11-06: a recording that is stopped mid-write leaves a 0-byte trailing file
    # (H:/dr/140hz_2g_14mm_P_1fps/..._2889.tiff). Drop trailing empties only: dropping one
    # from the middle would shift every later frame index without anyone noticing.
    trailing = 0
    while len(files) > trailing and files[-1 - trailing].stat().st_size == 0:
        trailing += 1
    if trailing:
        print('WARNING: dropping %d trailing 0-byte frame(s): %s'
              % (trailing, [f.name for f in files[len(files) - trailing:]]), flush=True)
        files = files[:len(files) - trailing]
    # 2026-11-08: do NOT stat every frame here. On the USB/exFAT archive this costs
    # ~3.3 ms/file, i.e. 30 s for 9000 frames, and every shard used to repeat it.
    # A 0-byte file in the middle now fails when fast_tiff tries to read it, which is
    # still a hard error (never a silent index shift). Set strict=True for tooling that
    # needs the old preflight check.
    if strict:
        for f in files:
            if f.stat().st_size == 0:
                raise IOError('0-byte frame in the middle of the sequence: ' + str(f))
    return files
