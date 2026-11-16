#pragma once
#include "bolt_common.h"

// LiLiCorr drafter (DFlash backbone + candidate lattice). Embeddings and the LM head are the target's.
// Every call tolerates a null Draft, so the engine runs without a drafter checkpoint.
struct Draft;
// nseq independent contexts (one per sequence); every call below names the sequence.
Draft* draft_open(const char* dir, const uint16_t* embed, const TQ8& head, int nseq);
void draft_reset(Draft* d, int seq);
// Absorb the target features of positions [pos0, pos0 + n): cap holds n rows of six concatenated residuals.
void draft_ingest(Draft* d, int seq, const float* cap, int pos0, int n, cudaStream_t s);
// The kept rows of several sequences' last block at once: cap holds the whole block; sequence seqs[i] keeps n[i] rows from
// row0[i] of it, at positions pos0[i]...; the last of them is its anchor.
void draft_ingest_multi(Draft* d, int nrows, const float* cap, int nseq, const int* seqs, const int* row0, const int* n, const int* pos0, cudaStream_t s);
// Seven draft tokens for the block that starts with `anchor` at position `pos`. Returns how many (0: none).
int draft_propose(Draft* d, int seq, int anchor, int pos, int* out, cudaStream_t s);
// How a stream's target rows are drawn: the lane core's keyed sampler at `position` for the first draft's row, the next
// position for each draft after it. temperature 0 (or no draws at all) is greedy.
struct BoltDraw { uint64_t seed; uint64_t position; float temperature; float top_p; };
// The same for several sequences at once (out is [n][8]; got[i] is 7, or 0 when stream i's context is not at pos[i]).
// With `draws`, stream i's drafts are drawn as draws[i] would draw them (the walk in bolt_draft.cu); null walks greedily.
void draft_propose_multi(Draft* d, int n, const int* seqs, const int* anchors, const int* pos, int* out, int* got, cudaStream_t s, const BoltDraw* draws = nullptr);
// The positions before `upto` will not be ingested (outside the sliding window); only the position advances.
void draft_skip(Draft* d, int seq, int upto);
// The drafter's context for prompt-cache snapshots: both K/V rings of every layer and the anchor, as one device blob.
size_t draft_blob_bytes();
void draft_save(Draft* d, int seq, void* dst, cudaStream_t s);
void draft_restore(Draft* d, int seq, const void* src, int clen, cudaStream_t s);
int draft_clen(Draft* d, int seq);
void draft_prof_report();
