import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const sourcePath = "D:/DSR_301m-/outputs/Team_Meeting_Data_Collection_thu_thap_xu_ly_v2.pptx";
const outDir = "D:/DSR_301m-/.codex_ppt/collection_processing_render_v2";

await fs.mkdir(outDir, { recursive: true });

const presentation = await PresentationFile.importPptx(await FileBlob.load(sourcePath));
const snapshot = await presentation.inspect({
  kind: "deck,slide,textbox,shape,notes,layout",
  maxChars: 50000,
});
await fs.writeFile(path.join(outDir, "snapshot.ndjson"), snapshot.ndjson, "utf8");

const montage = await presentation.export({ format: "png", montage: true, scale: 1 });
await fs.writeFile(path.join(outDir, "montage.png"), new Uint8Array(await montage.arrayBuffer()));

for (const line of snapshot.ndjson.split(/\r?\n/).filter(Boolean)) {
  const record = JSON.parse(line);
  if (record.kind !== "slide") continue;
  const slide = presentation.resolve(record.id);
  const preview = await slide.export({ format: "png", scale: 1 });
  await fs.writeFile(
    path.join(outDir, `slide-${String(record.slide).padStart(2, "0")}.png`),
    new Uint8Array(await preview.arrayBuffer()),
  );
}

console.log(outDir);
