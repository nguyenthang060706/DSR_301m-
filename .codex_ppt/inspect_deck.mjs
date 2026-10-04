import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const sourcePath = "D:/DSR_301m-/Team_Meeting_Data_Collection.pptx";
const outDir = "D:/DSR_301m-/.codex_ppt/inspect";

await fs.mkdir(outDir, { recursive: true });

const presentation = await PresentationFile.importPptx(await FileBlob.load(sourcePath));
const snapshot = await presentation.inspect({
  kind: "deck,slide,textbox,shape,image,table,chart,notes,thread,layout",
  maxChars: 50000,
});
await fs.writeFile(path.join(outDir, "snapshot.ndjson"), snapshot.ndjson, "utf8");

const montage = await presentation.export({ format: "png", montage: true, scale: 1 });
await fs.writeFile(path.join(outDir, "montage.png"), new Uint8Array(await montage.arrayBuffer()));

for (const slide of presentation.slides) {
  const preview = await slide.export({ format: "png", scale: 1 });
  await fs.writeFile(
    path.join(outDir, `slide-${String(slide.index + 1).padStart(2, "0")}.png`),
    new Uint8Array(await preview.arrayBuffer()),
  );
}

console.log(`Wrote inspection output to ${outDir}`);
