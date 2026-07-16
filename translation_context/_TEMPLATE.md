# Lecture Context (template)

Copy this file to `translation_context/<name>.md`, where `<name>` matches the
transcript stem (e.g. `transcript/arabic/lesson_01.txt` → `translation_context/lesson_01.md`).
The translation stage loads it automatically via `load_optional_context("<name>")`.

Files whose name begins with `_` (like this one) are ignored by the loader.

This context is used **only** to resolve ambiguity and to keep terminology,
names, and translation choices consistent. It must never be inserted into the
translation. Do not add any fact, ruling, evidence, or wording that is absent
from the Arabic source.

---

- Speaker: <e.g. Shaykh Salih al-Fawzan>
- Work being explained: <e.g. Kitab al-Tawhid>
- Topic: <e.g. Seeking assistance and seeking rescue>
- Register: <e.g. Formal lecture Arabic>
- Preferred terminology:
  - التوحيد → tawhid
  - الشرك → shirk
  - الاستعانة → seeking assistance
  - الاستغاثة → seeking rescue
- Known ASR issues:
  - The transcript often writes الاستغاثة as الاستعاذة.
