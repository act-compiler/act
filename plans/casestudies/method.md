# Method and limits

- Each case study read the spec pages `taidl/docs/*.md` and both examples first, and the grammars or front end only to settle a doubt; such reads are listed under doc gaps.
- Every sketch is a hand-written ISA specification; none has run through the front end, since the front end needs the `antlr4` runtime, which is not installed, and the repos stay read-only.
- All ten studies were written by the track lead in one session; no per-accelerator sub-agents or model mix were available.
- Only a sketch that ran through the front end counts as a site for a language extension; until C11 runs the sketches, the counts in `gaps.md` are documented sites.
- No Ascend study is made; the queued study is NVDLA, a descriptor-driven engine with public register documentation (C12).
- Eyeriss and CGRA have no public ISA; they are invented command sets and count zero.
- The Hopper sketch, with the five `wgmma` immediates and the async group, is the GPU tensor-core description.
- URLs on github.com, dl.acm.org, and hiascend.com were not reachable from this session and are cited from their public locations unverified; all other URLs returned HTTP 200 or 202.
- Verdicts judge the language as the spec states it, not what TAIDL-TO or ACT-Backend support today.
