"""Fetch the pinned study files and verify upstream SHA-256 before use."""
import argparse, json, urllib.request
from pathlib import Path
from run_provenance import file_digest
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    manifest=json.loads((ROOT/'provenance/datasets.json').read_text())
    files=manifest['files']+manifest.get('metadata_files',[])
    for r in files:
        out=ROOT/r['local_path']
        if out.exists() and file_digest(out)==r['sha256']:
            print('VERIFIED',r['local_path'],flush=True);continue
        if a.verify_only:raise SystemExit('Missing or changed input: '+r['local_path'])
        out.parent.mkdir(parents=True,exist_ok=True);temp=out.with_suffix(out.suffix+'.download')
        with urllib.request.urlopen(r['url'],timeout=60) as response,temp.open('wb') as handle:
            while block:=response.read(1024*1024):handle.write(block)
        if file_digest(temp)!=r['sha256']:raise SystemExit('Downloaded file checksum mismatch: '+str(temp))
        temp.replace(out);print('DOWNLOADED AND VERIFIED',r['local_path'],flush=True)

if __name__=='__main__':main()
