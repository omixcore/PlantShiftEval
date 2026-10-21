"""Structural citation audit plus Crossref metadata checks for the main manuscript."""
from __future__ import annotations
import argparse, difflib, json, re, time, unicodedata, urllib.error, urllib.parse, urllib.request
from pathlib import Path

YEAR_RE=re.compile(r"\b(?:19|20)\d{2}\b")
PUBLISHER_TITLE_OVERRIDES={"10.1145/3371158.3371196":{"title":"PlantDoc: A Dataset for Visual Plant Disease Detection","source":"https://doi.org/10.1145/3371158.3371196","reason":"Crossref exposes the short label PlantDoc; the official ACM record confirms the full article title, authors, year and pages 249–253."}}
DOI_RE=re.compile(r"10\.\d{4,9}/[^\s]+",re.I)
CITE_RE=re.compile(r"\[([0-9][0-9,\s–-]*)\]")

def parse_reference_lines(text):
    if "## References" not in text: raise ValueError("Missing ## References heading")
    block=text.split("## References",1)[1]
    refs={}
    for line in block.splitlines():
        m=re.match(r"^\s*(\d+)\.\s+(.+?)\s*$",line)
        if m: refs[int(m.group(1))]=m.group(2)
    return refs

def parse_citations(text,nrefs):
    body=text.split("## References",1)[0]
    cited=set(); invalid=[]
    for match in CITE_RE.finditer(body):
        raw=match.group(1).strip()
        if not re.fullmatch(r"\d+(?:\s*(?:,|–|-)\s*\d+)*",raw): continue
        for part in re.split(r"\s*,\s*",raw):
            nums=re.split(r"\s*[–-]\s*",part)
            if len(nums)==1: values=[int(nums[0])]
            elif len(nums)==2 and int(nums[0])<=int(nums[1]): values=list(range(int(nums[0]),int(nums[1])+1))
            else: invalid.append(raw); continue
            for value in values:
                cited.add(value)
                if value<1 or value>nrefs: invalid.append(raw)
    return sorted(cited),sorted(set(invalid))

def doi_from_ref(ref):
    m=DOI_RE.search(ref)
    return m.group(0).rstrip(".,;)]}>") if m else None

def normalize(s):
    s=unicodedata.normalize("NFKD",str(s)).encode("ascii","ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+"," ",s).strip()

def title_from_ref(ref):
    # In this reference style author initials are unpunctuated; the final author/title
    # boundary is the last capitalized sentence boundary before the journal/proceedings.
    end=re.search(r"\.\s+(?:\*|In:|arXiv|https?://)",ref)
    if not end: return None,None
    prefix=ref[:end.start()]
    boundaries=list(re.finditer(r"\.\s+(?=[A-Z])",prefix))
    if not boundaries: return None,None
    boundary=boundaries[-1]
    title=ref[boundary.end():end.start()].strip()
    authors=re.sub(r"^\s*\d+\.\s*","",ref[:boundary.start()]).strip()
    return title,authors

def first_author_surname(author_text):
    first=author_text.split(",",1)[0].strip()
    tokens=first.split()
    if len(tokens)<2: return normalize(first)
    return normalize(" ".join(tokens[:-1]))

def ref_year(ref):
    m=YEAR_RE.search(ref)
    return int(m.group(0)) if m else None

def get_crossref(doi,timeout=20):
    url="https://api.crossref.org/works/"+urllib.parse.quote(doi,safe="")
    request=urllib.request.Request(url,headers={"User-Agent":"PlantShiftEval/1.0 (https://github.com/omixcore/PlantShiftEval)"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request,timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))["message"],None
        except urllib.error.HTTPError as exc:
            if exc.code==404: return None,"not_found"
            if exc.code not in (429,500,502,503,504): return None,f"http_{exc.code}"
            time.sleep(1.0*(attempt+1))
        except Exception as exc:
            if attempt==2: return None,type(exc).__name__+": "+str(exc)
            time.sleep(1.0*(attempt+1))
    return None,"request_failed"

def source_audit_record(audit_text, number, reference):
    """Accept DOI-less records only when the separate audit documents a matching official source URL and completed metadata check."""
    reference_urls=[u.rstrip(".,;)]}") for u in re.findall(r"https?://[^\s]+",reference)]
    for line in audit_text.splitlines():
        if not line.startswith("|"):
            continue
        cells=[part.strip() for part in line.strip().strip("|").split("|")]
        if len(cells)<5:
            continue
        try:
            row_number=int(cells[0])
        except (ValueError,TypeError):
            continue
        if row_number!=number:
            continue
        metadata_note=cells[2]
        source_cell=cells[4]
        source_urls=re.findall(r"\]\((https?://[^)]+)\)",source_cell)
        same_url=next((u.rstrip(".,;)]}") for u in source_urls if u.rstrip(".,;)]}") in reference_urls),None)
        note=metadata_note.lower()
        documented_check=any(term in note for term in ("verified","official proceedings link","official link","corrected preprint year"))
        if same_url and documented_check:
            return {"source_audit_verified":True,"source_audit_metadata_note":metadata_note,"source_audit_url":same_url.rstrip(".,;)]}")}
        return None
    return None

def metadata_check(number,ref,verify_crossref=True):
    doi=doi_from_ref(ref); title,authors=title_from_ref(ref); year=ref_year(ref)
    row={"number":number,"reference":ref,"title_parsed":title,"doi":doi,"cited_year":year,
         "url_present":bool(re.search(r"https?://",ref)),"status":"not_checked"}
    if not doi:
        row["status"]="url_present_no_doi" if row["url_present"] else "missing_doi_and_url"
        return row
    if not verify_crossref:
        row["status"]="doi_present_not_resolved"
        return row
    data,error=get_crossref(doi)
    if error:
        row["status"]="crossref_"+error
        return row
    ctitle=(data.get("title") or [""])[0]
    row["crossref_title"]=ctitle
    override=PUBLISHER_TITLE_OVERRIDES.get(doi.lower())
    expected_title=override["title"] if override else ctitle
    row["title_similarity"]=difflib.SequenceMatcher(None,normalize(title or ""),normalize(expected_title)).ratio()
    if override:
        row["publisher_verified_title"]=override["title"]
        row["publisher_record"]=override["source"]
        row["crossref_metadata_note"]=override["reason"]
    authors_data=data.get("author") or []
    if authors_data:
        fam=normalize(authors_data[0].get("family",""))
        cited=first_author_surname(authors or "")
        row["cited_first_author_surname"]=cited
        row["crossref_first_author_family"]=fam
        row["first_author_match"]=bool(cited and fam and (cited==fam or cited in fam or fam in cited))
    else:
        row["first_author_match"]=None
    date_parts=(data.get("published-print") or data.get("published-online") or data.get("issued") or {}).get("date-parts") or []
    pub_year=int(date_parts[0][0]) if date_parts and date_parts[0] else None
    row["crossref_year"]=pub_year
    row["year_match"]=None if year is None or pub_year is None else abs(year-pub_year)<=1
    flags=[]
    if row["title_similarity"]<0.78: flags.append("title_mismatch")
    if row["first_author_match"] is False: flags.append("first_author_mismatch")
    if row["year_match"] is False: flags.append("year_mismatch")
    row["flags"]=flags
    row["status"]="needs_manual_review" if flags else ("publisher_title_override_crossref_abbreviated" if override else "crossref_match")
    return row

def audit(root:Path,verify_crossref=True):
    manuscript=root/"manuscript/PlantShiftEval_corrected_manuscript.md"
    audit_file=root/"manuscript/reference_audit.md"
    text=manuscript.read_text(encoding="utf-8")
    refs=parse_reference_lines(text)
    expected=list(range(1,len(refs)+1))
    structural=[]
    if sorted(refs)!=expected: structural.append("Reference numbering is not contiguous from 1")
    cited,invalid=parse_citations(text,len(refs))
    missing=sorted(set(refs)-set(cited))
    uncited=missing
    out_rows=[]
    for number,ref in refs.items():
        out_rows.append(metadata_check(number,ref,verify_crossref))
        if verify_crossref: time.sleep(0.12)
    if invalid: structural.append("Invalid/out-of-range citation expressions: "+", ".join(invalid))
    if uncited: structural.append("Uncited references: "+", ".join(map(str,uncited)))
    audit_text=""
    if not audit_file.exists(): structural.append("Missing reference_audit.md")
    else:
        audit_text=audit_file.read_text(encoding="utf-8")
        audit_count=len(re.findall(r"^\|\s*\d+\s*\|",audit_text,re.M))
        if audit_count!=len(refs): structural.append(f"Reference audit table rows {audit_count} do not match manuscript references {len(refs)}")
    for row in out_rows:
        if row["status"]=="url_present_no_doi":
            source_record=source_audit_record(audit_text,row["number"],row["reference"])
            if source_record:
                row.update(source_record)
                row["status"]="source_record_verified_no_doi"
    dois=[r["doi"].lower() for r in out_rows if r.get("doi")]
    duplicate_dois=sorted({d for d in dois if dois.count(d)>1})
    if duplicate_dois: structural.append("Duplicate DOI strings: "+", ".join(duplicate_dois))
    flagged=[r for r in out_rows if r["status"]=="needs_manual_review"]
    doi_checked=sum(bool(r.get("doi")) and ("crossref_title" in r or "publisher_verified_title" in r) for r in out_rows)
    url_only=[r for r in out_rows if r.get("url_present") and not r.get("doi")]
    url_only_verified=[r for r in url_only if r["status"]=="source_record_verified_no_doi"]
    url_only_unverified=[r for r in url_only if r["status"]!="source_record_verified_no_doi"]
    unresolved=[r for r in out_rows if r["status"].startswith("crossref_") and r["status"]!="crossref_match"]
    manual_review=flagged+url_only_unverified+unresolved+[r for r in out_rows if r["status"]=="missing_doi_and_url"]
    result={"schema_version":1,"manuscript":str(manuscript.relative_to(root)),"reference_count":len(refs),
        "citation_count":len(cited),"cited_reference_numbers":cited,"uncited_reference_numbers":uncited,
        "structural_errors":structural,"crossref_requested":verify_crossref,
        "doi_references_checked":doi_checked,"crossref_matched":sum(r["status"]=="crossref_match" for r in out_rows),
        "url_only_reference_numbers":[r["number"] for r in url_only],
        "source_record_verified_no_doi_reference_numbers":[r["number"] for r in url_only_verified],
        "url_only_unverified_reference_numbers":[r["number"] for r in url_only_unverified],
        "manual_review_reference_numbers":sorted({r["number"] for r in manual_review}),
        "metadata_flags":flagged,"reference_records":out_rows,
        "status":"failed" if structural else ("needs_manual_review" if manual_review else "passed")}
    out=root/"report/additional_baselines";out.mkdir(parents=True,exist_ok=True)
    (out/"bibliography_crossref_audit.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    lines=["# Bibliography metadata audit","",f"- Status: {result['status']}",f"- References: {len(refs)}",
        f"- In-text cited reference numbers: {len(cited)}",f"- DOI records with Crossref metadata retrieved: {result['doi_references_checked']}",
        f"- Exact Crossref title/author/year matches: {result['crossref_matched']}",
        f"- References with source URLs but no DOI: {len(url_only)}",
        f"- DOI-less references with documented source-record verification: {len(url_only_verified)}",
        f"- DOI-less references still needing source verification: {len(url_only_unverified)}",
        f"- References requiring manual review: {len(result['manual_review_reference_numbers'])}",
        f"- Metadata flags: {len(flagged)}",""]
    lines += ["| No. | Cited title | DOI | Crossref title | Similarity | Author/year checks | Status |",
              "|---:|---|---|---|---:|---|---|"]
    for r in out_rows:
        title=(r.get("title_parsed") or "").replace("|","/")
        ct=(r.get("crossref_title") or "").replace("|","/")
        sim=f"{r['title_similarity']:.3f}" if "title_similarity" in r else "—"
        checks=[]
        if "first_author_match" in r: checks.append("author="+str(r["first_author_match"]))
        if "year_match" in r: checks.append("year="+str(r["year_match"]))
        lines.append(f"| {r['number']} | {title} | {r.get('doi') or '—'} | {ct or '—'} | {sim} | {', '.join(checks) or 'not available'} | {r['status']} |")
    if structural: lines += ["","## Structural errors",""]+["- "+x for x in structural]
    if manual_review:
        lines += ["","## Manual review required",""]+[f"- Reference {r['number']}: status={r['status']}; cited title: {r.get('title_parsed')}; DOI: {r.get('doi') or 'none'}; URL present: {r.get('url_present')}; Crossref title: {r.get('crossref_title') or 'not retrieved'}; flags: {', '.join(r.get('flags',[])) or 'none'}" for r in sorted({r['number']:r for r in manual_review}.values(),key=lambda x:x['number'])]
    lines += ["","Crossref similarity is a triage signal, not a substitute for reading the publisher record. DOI-less preprints and proceedings are not counted as Crossref matches; they are marked source_record_verified_no_doi only when the separate reference audit documents a completed metadata check and the same official source URL. Otherwise they remain in the manual-review list.",""]
    (out/"bibliography_crossref_audit.md").write_text("\n".join(lines),encoding="utf-8")
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument("--skip-crossref",action="store_true");a=p.parse_args()
    r=audit(a.root,verify_crossref=not a.skip_crossref)
    print(json.dumps({"status":r["status"],"references":r["reference_count"],"cited":r["citation_count"],
        "structural_errors":r["structural_errors"],"metadata_flags":len(r["metadata_flags"]),
        "output":"report/additional_baselines/bibliography_crossref_audit.json"},indent=2))
    if r["structural_errors"]: raise SystemExit(1)
if __name__=="__main__": main()
