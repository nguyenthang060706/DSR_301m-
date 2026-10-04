import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const SKILL_DIR = "C:/Users/LENOVO/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations";
const workspaceDir = "D:\\DSR_301m-";
const sourcePath = path.join(workspaceDir, "outputs", "Team_Meeting_Data_Collection_dieu_chinh.pptx");
const outputDir = path.join(workspaceDir, "outputs");
const stagingDir = path.join(workspaceDir, ".codex_ppt", "staging");
const receiptDir = path.join(workspaceDir, ".codex-finalizer");
const finalPath = path.join(outputDir, "Team_Meeting_Data_Collection_thu_thap_xu_ly_v2.pptx");
const RUNTIME_PYTHON = "C:/Users/LENOVO/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe";

process.env.RUNTIME_NODE_MODULES = "C:/Users/LENOVO/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules";
process.env.RUNTIME_BIN_DIR = "C:/Users/LENOVO/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override";

const { finalizePresentation } = await import(
  new URL(`file:///${path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs").replaceAll("\\", "/")}`).href
);

await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(stagingDir, { recursive: true });
await fs.mkdir(receiptDir, { recursive: true });

const presentation = await PresentationFile.importPptx(await FileBlob.load(sourcePath));

const updates = [
  [
    "sh/3ah8rqlg",
    "Hiện trạng: đã có 15.000 ảnh công khai, đã map nhãn và chia Train/Dev cố định\nNguồn công khai: Kaggle Recyclable and Household Waste 30 lớp, hoặc TrashBusters làm phương án thay thế\nXử lý: tạo label_map.csv, gộp 30 lớp mịn về 3 nhóm quyết định, kiểm tra rò rỉ Train/Dev\nChia dữ liệu: 12.750 ảnh Train và 2.250 ảnh Dev/Holdout, tách theo lớp và domain\nDữ liệu canteen thật: chưa thu thập xong, đang chuẩn bị quy trình chụp và gán nhãn",
  ],
  [
    "sh/w72947yt",
    "Trước khi chụp: viết docs/label_guide.md, quy định rõ ca ranh giới như ly nhựa dính trà sữa hoặc hộp xốp dính dầu thuộc nhóm \"Còn lại\"\nKhi gán nhãn: 2 người gán độc lập 100 ảnh thử, yêu cầu Cohen's kappa >= 0,70\nSau khi gán: ảnh mơ hồ được gắn cờ riêng và không dùng để tinh chỉnh mô hình",
  ],
  [
    "sh/k7mxovud",
    "Thu thập canteen: chụp top-down trên khay hoặc bàn đồng nhất, một vật chính mỗi ảnh, thay đổi có kiểm soát về ánh sáng, góc xoay và độ bẩn\nMetadata: ghi cluster_id gồm ngày chụp, ca ăn và vật phẩm cụ thể để tránh rò rỉ dữ liệu\nXử lý sau thu thập: chia Val/Test/Few-shot theo cụm hoặc ngày, không chia ngẫu nhiên từng ảnh",
  ],
  [
    "sh/98rqt4r6",
    "Rủi ro chính: hiện chưa có dữ liệu canteen thật nên kết quả hiện tại mới dựa trên dữ liệu công khai đã xử lý\nNếu canteen từ chối cho chụp: xin phép ngay tuần 1, dự phòng bằng cách mua đồ ăn canteen về phòng thực nghiệm và dựng góc chụp mô phỏng\nNếu Test canteen quá nhỏ: dùng Cluster Bootstrap theo cluster_id và báo cáo effect size\nNếu trễ ở mốc Go/No-go tuần 5: loại M7 và Canteen-C, dồn thời gian cho Tier B, few-shot và đo trên thiết bị biên",
  ],
  [
    "sh/zedcfa9g",
    "Đã làm: xử lý dữ liệu công khai 15.000 ảnh, tạo label_map.csv và chia Train/Dev cố định\nTuần 1-2: xin phép canteen, viết label_guide.md, chụp thử khoảng 100 ảnh và kiểm tra kappa >= 0,70\nTuần 3-4: thu thập pool ảnh cốt lõi và Test canteen, ghi metadata theo cluster_id\nTuần 5 Go/No-go: đóng băng Test canteen >= 600 ảnh, kappa >= 0,70, khóa pool ảnh cốt lõi",
  ],
];

for (const [id, text] of updates) {
  presentation.resolve(id).text = text;
}

const candidatePath = path.join(stagingDir, "candidate_thu_thap_xu_ly.pptx");
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

await finalizePresentation({
  explicitTotalSlideCount: 7,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
  workspaceDir,
  candidatePath,
  finalPath,
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu",
    "9144000,5143500",
    "--validate-bullet-geometry",
    "--validate-heading-fit",
  ],
  verifyArtifactToolImport: true,
  receiptPath: path.join(receiptDir, "Team_Meeting_Data_Collection_thu_thap_xu_ly_v2.validation.json"),
});

console.log(finalPath);
