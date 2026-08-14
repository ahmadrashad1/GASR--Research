# module-4-5-exp.js — Documentation

## Purpose
- Generates a technology reference Word document (Module 4 & 5) describing algorithms and design choices for a 4D surgical-scene FYP.
- Uses the `docx` library to build a styled .docx file programmatically.

## Entry point
- Top-level script; running `node module-4-5-exp.js` produces `M4_M5_Technology_Document.docx` in the same folder as the script.

## Key sections / helpers
- Design tokens: color and layout constants (e.g., `NAVY`, `TEAL`, `CW`).
- Typography helpers: `R`, `RB`, `RI`, `body`, `H1`, `H2`, `H3`, `PB`, `SP` for consistent paragraph and run styles.
- Layout helpers: `code()` (renders code block table), `infoBox()` (two-column info panel), `trow()` / `thead()` (table rows/headers).
- Flow builder: collects document `ch` array via `P(...)` convenience and composes full `Document`.
- Build step: `Packer.toBuffer(doc).then(buf => fs.writeFileSync(out, buf));` saves the document.

## Dependencies
- Node.js (v14+ recommended)
- npm package: `docx`

## Run instructions (in workspace folder)
1. Ensure Node.js is installed: `node -v`
2. Initialize and install dependency:

```powershell
cd /d D:\fyp\module-4-5-explanation
npm init -y
npm install docx
```

3. Run the script:

```powershell
node module-4-5-exp.js
```

4. Output: `M4_M5_Technology_Document.docx` will appear in the same folder.

## Notes & recommendations
- The script now writes output to the script directory (`__dirname`), making it cross-platform.
- If `docx` is missing, `node` will print a `Cannot find module 'docx'` error. Install via `npm install docx`.
- If you want a PDF instead, open the generated .docx in Word/LibreOffice and export as PDF.

## Quick checklist I performed
- Read the script and inspected contents.
- Patched output path to be workspace-relative.
- Created this documentation file.

If you want, I can now:
- Install `docx` and run the script here (requires network and Node). 
- Or run it locally in your environment — tell me which you prefer.
