from app.services.blog_pipeline.assembler import _trim_to_budget, word_count

para = lambda n: "\n\n".join([" ".join(["word"] * 50)] * n)

sections = [
    {"role": "hook",     "text": para(4)},
    {"role": "context",  "text": para(8)},
    {"role": "evidence", "text": para(7)},
    {"role": "practical","text": para(10)},
    {"role": "faq",      "text": para(5)},
    {"role": "closing",  "text": para(2)},
]
before = sum(word_count(s["text"]) for s in sections)
trimmed = _trim_to_budget(sections, budget=1400)
after = sum(word_count(s["text"]) for s in trimmed)

print(f"Before: {before} words")
print(f"After:  {after} words (budget=1400)")
for s in trimmed:
    print(f"  {s['role']:15s}: {word_count(s['text'])} words")
