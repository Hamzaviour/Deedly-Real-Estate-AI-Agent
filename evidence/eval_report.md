# Evaluation report

Generated 2026-11-09 13:20 UTC by `python -m evals.run_evals`. All listings and documents are synthetic.

## Retrieval (RAG)

Retriever: `HybridKnowledgeBase`. 30 questions the documents cover, 6 they do not.

| Metric | Score |
|---|---|
| Hit@1 (expected document ranked first) | 90% |
| Hit@4 (expected document in top 4) | 100% |
| Mean reciprocal rank | 0.944 |
| Off-topic questions correctly given no passages | 100% |

Same questions, each retriever on its own:

| Retriever | Hit@1 | Hit@4 | MRR | Off-topic rejection |
|---|---|---|---|---|
| LexicalKnowledgeBase | 90% | 97% | 0.933 | 67% |
| KnowledgeBase | 93% | 93% | 0.933 | 100% |
| HybridKnowledgeBase | 90% | 100% | 0.944 | 100% |

LexicalKnowledgeBase is the hosted (serverless) fallback. It matches words, so a question that shares one word with the documents can retrieve a passage; the agent's instructions then require it to say the passages do not answer the question.

Misses:

- "The home I wanted got sold, what should I do?": expected ['01_buying_process.md'], got ['06_advisory_policy.md', '01_buying_process.md', '02_title_verification.md', '03_financing_affordability.md']
- "What extra papers do I need when buying an apartment?": expected ['02_title_verification.md'], got ['03_financing_affordability.md', '01_buying_process.md', '02_title_verification.md', '02_title_verification.md']
- "Can you guarantee this home will go up in value?": expected ['06_advisory_policy.md'], got ['04_inspection_guide.md', '06_advisory_policy.md', '03_financing_affordability.md', '07_taxes_and_transfer_costs.md']

## Agent behaviour

Skipped: no ANTHROPIC_API_KEY or GROQ_API_KEY set.
