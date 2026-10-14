"""Validate only the new review report and its source-note mapping."""
from pathlib import Path
import json
import re

review = Path(__file__).resolve().parent.parent
report = review / "ANA_RAPOR.md"
content = report.read_text(encoding="utf-8-sig")
# Windows drive paths are already absolute; remove an unintended POSIX prefix.
normalized = content.replace("](/C:/", "](C:/")
if normalized != content:
    report.write_text(normalized, encoding="utf-8")
content = normalized
sections = re.split(r"(?m)^### B(\d{2}) — ", content)
numbers = [int(sections[i]) for i in range(1, len(sections), 2)]
required_fields = ("Tür/kanıt:", "Tetikleyici/etki:", "Çözüm:", "Kabul:", "Efor:", "Kaynak uzman:")
missing_fields = {}
for i in range(1, len(sections), 2):
    body = sections[i + 1]
    missing = [field for field in required_fields if field not in body]
    if missing:
        missing_fields[f"B{sections[i]}"] = missing
id_pattern = r"(?:[MVTD]\d{2}-\d{2}|M\d{2}|\d{2}-\d{2})"
note_ids = []
note_records = []
for note in sorted((review / "uzman-notlari").glob("*.md")):
    text = note.read_text(encoding="utf-8-sig")
    ids = re.findall(rf"(?m)^#{{2,3}} ({id_pattern}) —", text)
    note_ids.extend(ids)
    note_records.append({"note": str(note), "findings": len(ids)})
mapped_ids = []
for match in re.finditer(r"\*\*Kaynak uzman:\*\* ([^\n]+)", content):
    mapped_ids.extend(re.findall(id_pattern, match.group(1)))
links = re.findall(r"\]\((C:/[^)]+)\)", content)
missing_links = []
for target in links:
    file_target = re.sub(r":\d+$", "", target)
    if not Path(file_target).is_file():
        missing_links.append(target)
result = {
    "report": str(report),
    "bytes": report.stat().st_size,
    "findings": len(numbers),
    "sequential_01_to_43": numbers == list(range(1, 44)),
    "missing_required_fields": missing_fields,
    "expert_notes": len(note_records),
    "original_expert_findings": len(note_ids),
    "mapped_expert_findings": len(mapped_ids),
    "unmapped_findings": sorted(set(note_ids) - set(mapped_ids)),
    "unknown_mapping_ids": sorted(set(mapped_ids) - set(note_ids)),
    "duplicate_mapping_ids": sorted({item for item in mapped_ids if mapped_ids.count(item) != 1}),
    "local_links_checked": len(links),
    "missing_local_links": missing_links,
    "utf8_replacement_char_found": "\ufffd" in content,
    "note_records": note_records,
}
result["passed"] = (
    result["sequential_01_to_43"] and not missing_fields
    and len(note_records) == 10 and len(note_ids) == len(mapped_ids) == 53
    and not result["unmapped_findings"] and not result["unknown_mapping_ids"]
    and not result["duplicate_mapping_ids"] and not missing_links
    and not result["utf8_replacement_char_found"]
)
(review / "kanitlar" / "rapor-son-kontrol.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps({key: value for key, value in result.items() if key != "note_records"}, ensure_ascii=False, indent=2))
raise SystemExit(0 if result["passed"] else 1)
