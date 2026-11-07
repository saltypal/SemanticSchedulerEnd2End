"""Framework-free inference for the scene93/context/temporal study.
Training exports exact dense, GRU, attention or numerical-tree artifacts here.
"""
import hashlib
import json
import math
import numpy as np

KINDS=('linear','mlp','classifier','deepsets','temporal','attention','trees')


def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def fit_normalizer(episodes):
    x=np.concatenate([np.array([m.features for m in e.messages]) for e in episodes])
    median=np.zeros(387);mean=np.zeros(387);scale=np.ones(387)
    for i in list(range(186))+[378,379]:
        mi=(186+i) if i<93 else (279+i-93) if i<186 else 384 if i==378 else 385
        v=x[x[:,mi]>.5,i]
        if len(v):median[i]=np.median(v);mean[i]=np.mean(v);scale[i]=max(float(np.std(v)),1e-8)
    scale[scale<1e-7]=1.
    profiles={m.profile_id for e in episodes for m in e.messages}
    if len(profiles)!=1:raise ValueError('training profiles differ')
    profile=episodes[0].metadata.get('input_profile',{'schema':'fixture','sha256':next(iter(profiles))})
    if profile['sha256']!=next(iter(profiles)):raise ValueError('profile manifest mismatch')
    return {'profile_identity':profile,'median':median.tolist(),'mean':mean.tolist(),'scale':scale.tolist(),'width':387,'continuous':list(range(186))+[378,379]}


def normalized(messages,pre):
    x=np.array([m.features for m in messages],dtype=np.float32).reshape(-1,387)
    idx=np.asarray(pre['continuous']);mi=np.where(idx<93,186+idx,np.where(idx<186,279+idx-93,np.where(idx==378,384,385)))
    mean=np.asarray(pre['mean'],np.float32)[idx];scale=np.asarray(pre['scale'],np.float32)[idx];median=np.asarray(pre['median'],np.float32)[idx]
    x[:,idx]=(np.where(x[:,mi]>.5,x[:,idx],median)-mean)/scale
    return x


def arrays(snapshot,pre,include_temporal=True):
    if snapshot.profile_id!=pre['profile_identity']['sha256']:raise ValueError('runtime producer/profile mismatch')
    def tokens(messages,role,finished=None):
        x=normalized(messages,pre)
        if len(messages):
            ages=[max(0,snapshot.now-(finished[i] if finished is not None else m.arrival)) for i,m in enumerate(messages)]
            return np.column_stack([x,np.log1p(ages),np.full(len(messages),role)]).astype(np.float32)
        return np.empty((0,389),dtype=np.float32)
    ready=snapshot.ready
    r=tokens(ready,0.)
    h=tokens([m for m,t in snapshot.processed],1.,[t for m,t in snapshot.processed])
    def window_rows(windows):
        a=np.zeros((len(windows),3,389),np.float32);mask=np.zeros((len(windows),3),bool)
        waiting=snapshot.waiting_tokens;inflight=set(snapshot.in_flight_tokens);failed=set(snapshot.failed_tokens)
        flat=[m for w in windows for m in w[-3:]]
        norm=normalized(flat,pre);k=0
        for i,w in enumerate(windows):
            for j,m in enumerate(w[-3:]):
                a[i,j,:387]=norm[k];k+=1;a[i,j,387]=np.log1p(max(0,snapshot.now-m.arrival))
                a[i,j,388]=3. if m.token in failed else 2. if m.token in inflight else 0. if m.token in waiting else 1.;mask[i,j]=True
        return a,mask
    uh,um=window_rows(snapshot.ue_windows if include_temporal else ());active,am=window_rows(snapshot.active_windows if include_temporal else ())
    sizes=np.array([np.log1p(len(r)),np.log1p(len(h)),np.log1p(snapshot.attempts)],np.float32)
    def means(a):
        v=a.mean(0) if len(a) else np.zeros(389,dtype=np.float32)
        for i in pre['continuous']:
            mi=186+i if i<93 else 279+i-93 if i<186 else 384 if i==378 else 385
            mask=a[:,mi]>.5
            v[i]=a[mask,i].mean() if mask.any() else 0.
        return v
    qm,gm=means(r),means(h)
    vi=list(range(186))+[374,375,376,377,378,379]
    mi=list(range(186,372))+[380,381,382,383,384,385]
    distances=np.zeros((len(r),2),np.float32)
    for j,row in enumerate(r):
        if not len(h):continue
        common=(h[:,mi]>.5)&(row[mi]>.5);count=common.sum(1);good=count>0
        if good.any():
            squared=np.where(common,(h[:,vi]-row[vi])**2,0.).sum(1)
            distances[j]=[np.sqrt(squared[good]/count[good]).min(),1.]
    summary=np.concatenate([r,np.tile(qm,(len(r),1)),np.tile(gm,(len(r),1)),np.tile(sizes,(len(r),1)),distances],axis=1).astype(np.float32)
    return dict(ready=r,processed=h,ue=uh,ue_mask=um,active=active,active_mask=am,sizes=sizes,distances=distances,summary=summary)


def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-60,60)))
def softmax(x):
    z=np.exp(x-np.max(x,axis=-1,keepdims=True));return z/z.sum(axis=-1,keepdims=True)


def validate_artifact(obj, weights):
    """Reject malformed numerical artifacts at load time, before queue dispatch."""
    pre=obj["preprocessing"]
    for name in ("median","mean","scale"):
        value=np.asarray(pre.get(name),dtype=float)
        if value.shape!=(387,) or not np.isfinite(value).all():
            raise ValueError("invalid preprocessing "+name)
        if name=="scale" and np.any(value<=0):raise ValueError("positive scales required")
    columns=pre.get("continuous")
    if not isinstance(columns,list) or len(set(columns))!=len(columns) or any(
            type(value) is not int or value not in list(range(186))+[378,379] for value in columns):
        raise ValueError("invalid continuous feature registry")
    kind=obj["kind"]
    expected={}
    def dense(prefix,out,inp):
        expected[prefix+".weight"]=(out,inp);expected[prefix+".bias"]=(out,)
    def head(inp,out=1):
        dense("head.0",64,inp);dense("head.2",32,64);dense("head.4",out,32)
    if kind=="linear":dense("linear",1,1172)
    elif kind in ("mlp","classifier"):
        head(1172,4 if kind=="classifier" else 1)
        if kind=="classifier":
            cw=np.asarray(obj.get("class_weights"),dtype=float)
            if cw.shape!=(4,) or not np.isfinite(cw).all() or np.any(cw<=0):
                raise ValueError("positive finite classifier weights required")
    elif kind in ("deepsets","temporal"):
        dense("phi.0",32,389);dense("phi.2",32,32)
        head(554 if kind=="temporal" else 490)
        if kind=="temporal":
            expected.update({"gru.weight_ih":(96,37),"gru.weight_hh":(96,32),
                             "gru.bias_ih":(96,),"gru.bias_hh":(96,)})
    elif kind=="attention":
        dense("projection",32,389);head(426)
        for index in range(2):
            prefix="attention.layers."+str(index)
            expected[prefix+".self_attn.in_proj_weight"]=(96,32)
            expected[prefix+".self_attn.in_proj_bias"]=(96,)
            dense(prefix+".self_attn.out_proj",32,32)
            dense(prefix+".linear1",64,32);dense(prefix+".linear2",32,64)
            for norm in ("norm1","norm2"):
                expected[prefix+"."+norm+".weight"]=(32,)
                expected[prefix+"."+norm+".bias"]=(32,)
    elif kind=="trees":
        trees=obj.get("trees")
        if not isinstance(trees,list) or not trees:raise ValueError("numerical trees required")
        def inspect(node):
            if not isinstance(node,dict):raise ValueError("invalid numerical tree")
            if "leaf_value" in node:
                if not math.isfinite(node["leaf_value"]):raise ValueError("nonfinite tree leaf")
                return
            feature=node.get("split_feature")
            if (type(feature) is not int or not 0<=feature<1172 or
                    node.get("decision_type")!="<=" or not math.isfinite(node["threshold"]) or
                    node.get("missing_type","None") not in ("None","NaN")):
                raise ValueError("unsupported numerical tree split")
            inspect(node["left_child"]);inspect(node["right_child"])
        for tree in trees:inspect(tree)
    if set(weights)!=set(expected) or any(weights[name].shape!=shape for name,shape in expected.items()):
        raise ValueError("model weight keys/shapes do not match architecture")


class NumpyRanker:
    def __init__(self,artifact):
        obj=dict(artifact);checksum=obj.pop('sha256',None)
        if checksum!=digest(obj) or obj.get('schema')!='semantic_scene_ranker_v1':raise ValueError('artifact identity mismatch')
        if obj['kind'] not in KINDS or obj['preprocessing']['width']!=387:raise ValueError('unsupported model/profile')
        if obj.get('input_profile')!=obj['preprocessing'].get('profile_identity') or not obj.get('objective_version') or 'reward_version' not in obj:raise ValueError('model profile/objective binding missing')
        self.artifact=artifact;self.kind=obj['kind'];self.pre=obj['preprocessing'];self.w={k:np.asarray(v,dtype=float) for k,v in obj.get('weights',{}).items()}
        if any(not np.isfinite(v).all() for v in self.w.values()):raise ValueError('nonfinite model weights')
        validate_artifact(obj,self.w)
        self.fallbacks=0

    def linear(self,x,name):return x@self.w[name+'.weight'].T+self.w[name+'.bias']
    def mlp(self,x,prefix,layers):
        for j in range(layers):
            x=self.linear(x,f'{prefix}.{j*2}')
            if j<layers-1:x=np.maximum(x,0)
        return x
    def phi(self,x):return np.maximum(self.linear(np.maximum(self.linear(x,'phi.0'),0),'phi.2'),0)
    def gru(self,tokens,mask):
        encoded=self.phi(tokens.reshape(-1,389)).reshape(len(tokens),3,32)
        hidden=np.zeros((len(tokens),32))
        for t in range(3):
            # encoded32, timegap1, waiting/processed/inflight3, valid1 =37.
            status=tokens[:,t,388]
            inp=np.column_stack([encoded[:,t],tokens[:,t,387],status==0,status==1,status==2,mask[:,t]])
            gi=inp@self.w['gru.weight_ih'].T+self.w['gru.bias_ih'];gh=hidden@self.w['gru.weight_hh'].T+self.w['gru.bias_hh']
            ir,iz,inn=np.split(gi,3,axis=1);hr,hz,hn=np.split(gh,3,axis=1)
            r,z=sigmoid(ir+hr),sigmoid(iz+hz);n=np.tanh(inn+r*hn)
            proposed=(1-z)*n+z*hidden;hidden=np.where(mask[:,t,None],proposed,hidden)
        return hidden
    def layernorm(self,x,p):
        return (x-x.mean(-1,keepdims=True))/np.sqrt(x.var(-1,keepdims=True)+1e-5)*self.w[p+'.weight']+self.w[p+'.bias']
    def attention(self,tokens):
        x=self.linear(tokens,'projection')
        for k in range(2):
            p=f'attention.layers.{k}';att=p+'.self_attn'
            z=x@self.w[att+'.in_proj_weight'].T+self.w[att+'.in_proj_bias'];q,key,v=np.split(z,3,axis=-1)
            q=q.reshape(len(x),4,8).transpose(1,0,2);key=key.reshape(len(x),4,8).transpose(1,0,2);v=v.reshape(len(x),4,8).transpose(1,0,2)
            weights=softmax(q@key.transpose(0,2,1)/math.sqrt(8))
            out=(weights@v).transpose(1,0,2).reshape(len(x),32)
            out=self.linear(out,att+'.out_proj');x=self.layernorm(x+out,p+'.norm1')
            out=self.linear(np.maximum(self.linear(x,p+'.linear1'),0),p+'.linear2');x=self.layernorm(x+out,p+'.norm2')
        return x

    def raw(self,s):
        a=arrays(s,self.pre,self.kind=='temporal');r,h=a['ready'],a['processed'];n=len(r)
        if self.kind=='trees':
            def tree(node,row):
                if 'leaf_value' in node:return node['leaf_value']
                if node.get('decision_type')!='<=':raise ValueError('unsupported tree split')
                v=row[node['split_feature']];left=(node.get('default_left',True) if np.isnan(v) else float(v)<=float(node['threshold']))
                return tree(node['left_child'] if left else node['right_child'],row)
            return np.array([sum(tree(t,row) for t in self.artifact['trees']) for row in a['summary']])
        if self.kind=='linear':return self.linear(a['summary'],'linear').reshape(-1)
        if self.kind in ('mlp','classifier'):
            out=self.mlp(a['summary'],'head',3)
            return out if self.kind=='classifier' else out.reshape(-1)
        sizes=np.tile(a['sizes'],(n,1))
        if self.kind=='attention':
            z=self.attention(np.concatenate([r,h]))[:n];inputs=np.concatenate([r,z,sizes,a['distances']],axis=1)
        else:
            er=self.phi(r);eh=self.phi(h) if len(h) else np.zeros((0,32))
            qm=er.mean(0);gm=eh.mean(0) if len(eh) else np.zeros(32)
            inputs=np.concatenate([r,er,np.tile(qm,(n,1)),np.tile(gm,(n,1)),sizes,a['distances']],axis=1)
            if self.kind=='temporal':
                hh=self.gru(a['ue'],a['ue_mask']);uu=self.gru(a['active'],a['active_mask'])
                up=uu.mean(0) if len(uu) else np.zeros(32);inputs=np.concatenate([inputs,hh,np.tile(up,(n,1))],axis=1)
        return self.mlp(inputs,'head',3).reshape(-1)

    def scores(self,s):
        raw=self.raw(s)
        if self.kind=='classifier':
            q=softmax(raw);cw=np.asarray(self.artifact['class_weights']);q=q/cw;q=q/q.sum(1,keepdims=True)
            wait=np.array([s.now-m.arrival for m in s.ready])[:,None];delta=np.array([2.,4.,8.,16.])[None,:]
            def cost(w):z=w/delta;return z+np.maximum(z-1,0)
            return np.sum(q*(cost(wait+1)-cost(wait)),axis=1)
        return np.asarray(raw).reshape(-1)

    def select(self,s):
        if not s.ready:return None
        scores=self.scores(s)
        if scores.shape!=(len(s.ready),) or not np.isfinite(scores).all():
            self.fallbacks+=1;return s.ready[0].token
        return s.ready[int(np.argmax(scores))].token
