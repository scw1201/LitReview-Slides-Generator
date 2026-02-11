# Input Manifest Schema

Use this schema when exporting papers from Zotero MCP before running the generator.

## File Format

JSON object:

```json
{
  "collection": "VR-Obstacles",
  "items": [
    {
      "title": "Paper title",
      "authors": ["Author A", "Author B"],
      "institution": "Stanford University",
      "venue": "ICRA",
      "year": 2025,
      "doi": "10.0000/xxxx",
      "pdf_path": "/absolute/path/to/file.pdf"
    }
  ]
}
```

## Required fields

- `collection` (string)
- `items` (array)
- for each item: `pdf_path` (absolute path to local PDF)

## Optional fields

- `title`, `authors`, `institution`, `venue`, `year`, `doi`

The pipeline will fill missing values with `unknown`.
