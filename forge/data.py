"""Deterministic local character language-model data."""
import hashlib
from pathlib import Path
import numpy as np

class CharDataset:
    def __init__(self,text,context,validation_fraction=0.1):
        if not text or context<1 or not 0<validation_fraction<1: raise ValueError('invalid dataset configuration')
        self.text=text; self.context=context; self.chars=sorted(set(text))
        if len(self.chars)<2: raise ValueError('text needs at least two unique characters')
        self.stoi={c:i for i,c in enumerate(self.chars)}; self.itos={i:c for i,c in enumerate(self.chars)}
        encoded=np.asarray([self.stoi[c] for c in text],dtype=np.int64); cut=max(context+1,int(len(encoded)*(1-validation_fraction)))
        self.train=encoded[:cut]; self.validation=encoded[cut:]
        if len(self.validation)<=context: raise ValueError('validation split is shorter than context')
        self.fingerprint=hashlib.sha256(text.encode()).hexdigest()
    @classmethod
    def from_file(cls,path,context,validation_fraction=0.1):
        p=Path(path); return cls(p.read_text(encoding='utf-8'),context,validation_fraction)
    def batch(self,split,batch_size,step,seed=0):
        data={'train':self.train,'validation':self.validation}.get(split)
        if data is None: raise ValueError('split must be train or validation')
        rng=np.random.default_rng(seed+step); starts=rng.integers(0,len(data)-self.context,size=batch_size)
        return (np.stack([data[i:i+self.context] for i in starts]),np.stack([data[i+1:i+self.context+1] for i in starts]))
    def metadata(self):
        return {'kind':'character-language-model','sha256':self.fingerprint,'characters':len(self.chars),'total_tokens':len(self.text),'train_tokens':len(self.train),'validation_tokens':len(self.validation),'context':self.context}
