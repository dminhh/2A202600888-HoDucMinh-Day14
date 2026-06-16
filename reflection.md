# Day 14 — Reflection
## Evaluation Report & Failure Analysis

---

## 1. Benchmark Results Summary

Benchmark chạy trên golden dataset 20 QA pairs (domain: AI/RAG Pipeline), dùng mock agent với pre-written answers.

**Overall pass rate: 20%** (4/20 passed)

**Average scores:**

| Metric | Average | Min | Max | Std Dev |
|--------|---------|-----|-----|---------|
| Faithfulness | 0.44 | 0.17 | 0.70 | 0.16 |
| Relevance | 0.38 | 0.11 | 1.00 | 0.21 |
| Completeness | 0.55 | 0.33 | 0.81 | 0.14 |
| Overall Score | 0.46 | 0.25 | 0.73 | 0.11 |

**Score interpretation (theo bài giảng):**
- Bao nhiêu metrics ở Good (0.8–1.0): **0** — không có metric nào đạt ngưỡng tốt ở mức avg
- Bao nhiêu metrics ở Needs Work (0.6–0.8): **1** (Completeness avg = 0.55 tiệm cận ngưỡng này)
- Bao nhiêu metrics ở Significant Issues (< 0.6): **3** (Faithfulness, Relevance, Overall đều < 0.6)

**Failure type distribution:**

| Failure Type | Count | Percentage |
|--------------|-------|------------|
| hallucination | 5 | 31.3% |
| irrelevant | 6 | 37.5% |
| incomplete | 0 | 0% |
| off_topic | 5 | 31.3% |
| refusal | 0 | 0% |

> **Nhận xét:** Relevance thấp nhất (0.38) cho thấy agent thường trả lời không đúng trọng tâm câu hỏi.
> Đây là dấu hiệu của prompt ambiguity và intent mismatch. Faithfulness thấp (0.44) cho thấy agent
> dùng knowledge ngoài context nhiều hơn nên nhớ. Word-overlap heuristic strict hơn LLM-based metrics.

---

## 2. Top 3 Worst Failures — 5 Whys Analysis

Theo bài giảng: "Phân loại failure TRƯỚC KHI fix. Đừng fix từng failure riêng lẻ — CLUSTER rồi fix root cause."

---

### Failure 1

**Question:** What trade-offs exist between chunk overlap and retrieval quality?

**Agent Answer:** "Overlap helps continuity but increases storage and duplication. Too much overlap hurts precision."

**Scores:** Faithfulness: 0.27 | Relevance: 0.11 | Completeness: 0.35 | Overall: **0.25** (worst in dataset)

**5 Whys Analysis:**

| Level | Question | Answer |
|-------|----------|--------|
| Symptom | Vấn đề là gì? | Score tổng thấp nhất dataset (0.25), đặc biệt Relevance chỉ 0.11 |
| Why 1 | Tại sao Relevance thấp? | Answer không dùng từ trong câu hỏi ("trade-offs", "retrieval quality") |
| Why 2 | Tại sao answer không cover từ câu hỏi? | Agent tóm tắt quá ngắn, bỏ qua technical terms từ question |
| Why 3 | Tại sao agent tóm tắt ngắn? | Câu hỏi là hard-level với nhiều sub-concepts, agent chọn paraphrase thay vì answer trực tiếp |
| Why 4 | Root cause là gì? | Prompt không yêu cầu agent giữ lại terminology từ câu hỏi; câu hỏi hard không có few-shot example |

**Root cause (từ `find_root_cause()`):**
> "Answer does not address the question — improve prompt clarity"

**Bạn có đồng ý với root cause suggestion không? Tại sao?**
> Đồng ý một phần. Root cause đúng về mặt triệu chứng (relevance thấp = không address câu hỏi).
> Nhưng deeper root cause là thiếu few-shot examples cho hard-level queries và agent paraphrase thay
> vì giữ nguyên technical terminology. Fix prompt clarity là đúng hướng nhưng cần thêm few-shot.

**Proposed fix:**
> 1. Thêm system prompt instruction: "Always use the same technical terms as in the question."
> 2. Thêm 2–3 few-shot examples cho hard-level questions để agent học cách answer structured.

---

### Failure 2

**Question:** How does reranking improve RAG performance?

**Agent Answer:** "Reranking re-scores retrieved chunks to put the most relevant ones first, improving context precision."

**Scores:** Faithfulness: 0.23 | Relevance: 0.17 | Completeness: 0.62 | Overall: **0.34**

**5 Whys Analysis:**

| Level | Question | Answer |
|-------|----------|--------|
| Symptom | Vấn đề là gì? | Faithfulness=0.23 và Relevance=0.17 dù answer nghe có vẻ đúng |
| Why 1 | Tại sao Faithfulness thấp? | Answer dùng nhiều từ ("cross-encoder", "scores") không có trong context ngắn được cung cấp |
| Why 2 | Tại sao context không đủ? | Context chỉ 2 câu ngắn, không cover đầy đủ terminology của answer |
| Why 3 | Tại sao Relevance thấp dù answer đúng? | Word-overlap heuristic: "reranking", "improve", "RAG", "performance" khác với "reranker", "cross-encoder", "query" trong question |
| Why 4 | Root cause là gì? | Context quá ngắn và word-overlap metric không bắt semantic similarity → metric mismatch với quality thực tế |

**Root cause:**
> Đây là false negative của word-overlap heuristic: answer thực ra ĐÚNG nhưng metric đánh giá thấp
> vì paraphrase và không dùng đúng từ trong context. Root cause thực là metric limitation, không phải agent quality.

**Proposed fix:**
> 1. Augment context: cung cấp context đầy đủ hơn với nhiều từ liên quan (cross-encoder, re-scores, rank).
> 2. Dài hạn: thay word-overlap bằng embedding-based similarity hoặc LLM-as-Judge cho semantic matching.

---

### Failure 3

**Question:** What is hallucination in LLMs?

**Agent Answer:** "Hallucination is when an AI model generates false information that sounds confident but is not grounded in evidence."

**Scores:** Faithfulness: 0.23 | Relevance: 0.33 | Completeness: 0.55 | Overall: **0.37**

**5 Whys Analysis:**

| Level | Question | Answer |
|-------|----------|--------|
| Symptom | Vấn đề là gì? | Câu hỏi Easy nhưng score thấp (0.37), đặc biệt Faithfulness chỉ 0.23 |
| Why 1 | Tại sao Faithfulness thấp? | Answer dùng "AI model", "confident", "evidence" không có trong context |
| Why 2 | Tại sao agent dùng từ ngoài context? | Context nói "LLM hallucination", "false or unsupported facts", "RAG helps" — agent paraphrase sang wording khác |
| Why 3 | Tại sao agent paraphrase? | Agent có prior knowledge về hallucination và dùng nó thay vì grounding vào context |
| Why 4 | Root cause là gì? | RAG pipeline không force agent to cite context; agent ưu tiên fluency/naturalness hơn faithfulness |

**Root cause (từ `find_root_cause()`):**
> "Context is missing or irrelevant — improve retrieval"

**Bạn có đồng ý với root cause suggestion không? Tại sao?**
> Không hoàn toàn. Context không missing — agent đơn giản chọn không dùng nó. Root cause thực
> là thiếu "cite-from-context" instruction trong prompt, không phải retrieval kém.

**Proposed fix:**
> 1. Thêm instruction: "Answer ONLY using information in the provided context. Do not add external knowledge."
> 2. Thêm post-processing faithfulness check: so sánh answer tokens với context tokens trước khi return.

---

## 3. Failure Clustering

Theo bài giảng: "Fix 1 root cause giải quyết nhiều failures cùng lúc."

**Cluster Analysis:**

| Cluster | Root Cause | Failures in cluster | Priority |
|---------|-----------|--------------------:|----------|
| 1 — Relevance Gap | Agent không giữ terminology từ question; paraphrase làm giảm word-overlap | 6 failures (irrelevant) | **High** |
| 2 — Context Unfaithfulness | Agent dùng prior knowledge thay vì grounding vào context, không có cite-from-context instruction | 5 failures (hallucination) | **High** |
| 3 — Metric Mismatch | Word-overlap heuristic không bắt được paraphrase/semantic similarity | 5 failures (off_topic) | Medium |

**Nếu chỉ fix 1 cluster, bạn chọn cluster nào? Tại sao?**
> Chọn **Cluster 2 — Context Unfaithfulness** vì:
> 1. Hallucination là rủi ro cao nhất trong production (user trust và safety).
> 2. Fix đơn giản: thêm 1 instruction vào system prompt có thể giải quyết 5/16 failures ngay lập tức.
> 3. Cluster 3 (metric mismatch) cần nâng cấp infrastructure (LLM judge) — tốn resource hơn.
> 4. Cluster 1 cần redesign dataset + few-shot examples — công sức nhiều hơn.

---

## 4. Improvement Log (từ `generate_improvement_log`)

```
| Failure ID | Type | Root Cause | Suggested Fix | Status |
|------------|------|------------|---------------|--------|
| F001 | off_topic | Context is missing or irrelevant — improve retrieval | Implement hallucination checker to filter unsupported claims | Open |
| F002 | irrelevant | Answer does not address the question — improve prompt clarity | Improve prompt clarity and intent detection to address off-topic answers | Open |
| F003 | off_topic | Context is missing or irrelevant — improve retrieval | Add routing logic to better handle ambiguous or out-of-scope queries | Open |
| F004 | hallucination | Context is missing or irrelevant — improve retrieval | Review and fix | Open |
| F005 | off_topic | Answer does not address the question — improve prompt clarity | Review and fix | Open |
| F006 | irrelevant | Answer does not address the question — improve prompt clarity | Review and fix | Open |
| F007 | irrelevant | Answer does not address the question — improve prompt clarity | Review and fix | Open |
| F008 | off_topic | Answer is missing key information — increase context window or improve generation | Review and fix | Open |
| F009 | hallucination | Answer does not address the question — improve prompt clarity | Review and fix | Open |
| F010 | hallucination | Context is missing or irrelevant — improve retrieval | Review and fix | Open |
| F011 | irrelevant | Answer does not address the question — improve prompt clarity | Review and fix | Open |
| F012 | irrelevant | Answer does not address the question — improve prompt clarity | Review and fix | Open |
| F013 | hallucination | Answer does not address the question — improve prompt clarity | Review and fix | Open |
| F014 | off_topic | Answer does not address the question — improve prompt clarity | Review and fix | Open |
| F015 | hallucination | Context is missing or irrelevant — improve retrieval | Review and fix | Open |
| F016 | irrelevant | Answer does not address the question — improve prompt clarity | Review and fix | Open |
```

**3 improvement suggestions từ `generate_improvement_suggestions()`:**
1. Implement hallucination checker to filter unsupported claims
2. Improve prompt clarity and intent detection to address off-topic answers
3. Add routing logic to better handle ambiguous or out-of-scope queries

---

## 5. Regression Testing Strategy

### CI/CD Integration

**Câu 1: Khi nào chạy `run_regression()` trong production system?**
> Chạy `run_regression()` tại 3 trigger points:
> - **Pre-merge:** Trước mỗi PR merge vào main — so sánh feature branch vs main baseline.
> - **Post-prompt-change:** Sau mỗi lần thay đổi system prompt hoặc few-shot examples.
> - **Nightly CI:** Chạy tự động mỗi đêm trên golden dataset để phát hiện model drift sớm.
>
> Baseline được cập nhật mỗi sprint sau khi confirm improvements là intentional.

**Câu 2: Threshold regression 0.05 có phù hợp domain này không?**
> Với AI education assistant, threshold 0.05 là hợp lý cho Faithfulness và Completeness.
> Riêng Relevance nên strict hơn: threshold 0.03 vì Relevance thấp (0.38 avg) — drop thêm 0.05
> có thể kéo xuống mức critical (<0.33). Faithfulness có thể loose hơn đôi chút (0.07) vì
> word-overlap metric noisy, tránh false alarm quá nhiều.

**Câu 3: Khi phát hiện regression — block deployment hay chỉ alert?**
> **Faithfulness regression → BLOCK.** Hallucination trong production gây mất trust và có thể
> spread misinformation. Cost of blocking (delay release) < cost of hallucinating to users.
>
> **Relevance regression → ALERT + human review.** Relevance drop có thể do metric noise
> (paraphrase). Cần human review trước khi block để tránh false positives.
>
> **Completeness regression → ALERT only.** Thường do intentional tóm tắt, không critical.
> Monitor 3 days, block nếu downtrend tiếp tục.

**Câu 4: Eval pipeline nên chạy ở đâu trong CI/CD flow?**

```
Code change → [Unit Tests + Pytest Eval] → [Regression Check vs Baseline] → [Human Review (nếu borderline)] → Deploy
              (bước 1: nhanh, auto-block)    (bước 2: compare metrics)         (bước 3: edge cases)
```

---

## 6. Continuous Improvement Loop

Theo bài giảng: Evaluate → Analyze → Improve → Augment (add to benchmark) → lặp lại

**Sau lab hôm nay, 3 actions tiếp theo để improve agent:**

| Priority | Action | Metric sẽ improve | Expected impact |
|----------|--------|-------------------|-----------------|
| 1 | Thêm "cite-from-context only" instruction vào system prompt | Faithfulness ↑ | +0.15–0.20 (giảm hallucination cases từ 5 xuống 1–2) |
| 2 | Thêm terminology-preservation instruction + few-shot examples cho hard queries | Relevance ↑ | +0.10–0.15 (giảm irrelevant cases từ 6 xuống 2–3) |
| 3 | Nâng cấp context: mở rộng context string cho mỗi QAPair (200+ tokens thay vì 1–2 câu) | Faithfulness + Completeness ↑ | +0.10 mỗi metric |

**Failure cases cần thêm vào benchmark cho sprint tiếp theo:**
> 1. **Multi-hop reasoning:** "Nếu RAG dùng hybrid search và context window 128k, tối đa bao nhiêu chunks có thể inject?" — test khả năng kết hợp nhiều concepts.
> 2. **Contradictory context:** Provide 2 chunks mâu thuẫn nhau, xem agent handle thế nào — test faithfulness với conflicting evidence.
> 3. **Long-form expected answer:** Expected answer > 100 words — test completeness với câu hỏi cần giải thích chi tiết.

---

## 7. Framework Reflection

**Framework đã dùng trong lab:** RAGAS-inspired word-overlap heuristic

**Nếu dùng trong production, bạn sẽ chọn framework nào?**

Chọn **DeepEval + RAGAS** kết hợp:

| Tiêu chí | Lý do chọn |
|----------|------------|
| Focus phù hợp vì... | RAGAS cung cấp RAG-specific metrics (Context Recall, Precision, Faithfulness) đúng với use case RAG chatbot. DeepEval thêm safety metrics và hallucination detection |
| CI/CD integration vì... | DeepEval native pytest integration: `deepeval test run test_eval.py` trong GitHub Actions. Pass/fail = standard exit code, dễ block merge |
| Team workflow vì... | RAGAS dashboard cho product manager xem trend; DeepEval assertions cho developers thấy failure detail ngay trong terminal khi chạy test |

**Điểm yếu của word-overlap heuristic:**
> Không bắt được semantic similarity → high false negative rate (answer đúng nhưng score thấp).
> Không xử lý được paraphrase, synonym, hay multilingual content.
> Trong production, nên dùng LLM-as-Judge cho faithfulness và relevance, giữ word-overlap chỉ cho
> regression trend detection (nhanh, deterministic, không tốn API cost).