import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const SKILL_DIR = "C:/Users/LENOVO/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations";
const workspaceDir = "D:\\DSR_301m-";
const sourcePath = path.join(workspaceDir, "Team_Meeting_Data_Collection.pptx");
const stagingDir = path.join(workspaceDir, ".codex_ppt", "staging");
const receiptDir = path.join(workspaceDir, ".codex-finalizer");
const outputDir = path.join(workspaceDir, "outputs");
const finalPath = path.join(outputDir, "Team_Meeting_Data_Collection_dieu_chinh.pptx");
const RUNTIME_PYTHON = "C:/Users/LENOVO/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe";
const RUNTIME_NODE_MODULES = "C:/Users/LENOVO/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules";
const RUNTIME_BIN_DIR = "C:/Users/LENOVO/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override";

process.env.RUNTIME_NODE_MODULES = RUNTIME_NODE_MODULES;
process.env.RUNTIME_BIN_DIR = RUNTIME_BIN_DIR;

const { finalizePresentation } = await import(pathToFileURL(
  path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs"),
).href);

await fs.mkdir(stagingDir, { recursive: true });
await fs.mkdir(receiptDir, { recursive: true });
await fs.mkdir(outputDir, { recursive: true });

const presentation = await PresentationFile.importPptx(await FileBlob.load(sourcePath));

const replacements = [
  ["sh/547294r6", "Team Meeting: Data Collection", "Họp nhóm: Thu thập dữ liệu"],
  [
    "sh/7qp4be9c",
    "KD cho phân loại rác canteen trên thiết bị biên | Kế hoạch MASTER v9.5",
    "Knowledge Distillation cho phân loại rác canteen trên thiết bị biên, theo kế hoạch MASTER v9.5",
  ],
  [
    "sh/ozy1ofad",
    "Mục tiêu dữ liệu: 3 nhóm quyết định",
    "Mục tiêu dữ liệu: 3 nhóm quyết định",
  ],
  [
    "sh/wjy9sry9",
    "Phân loại theo quyết định bỏ rác thực tế tại canteen\nCần 2 loại dữ liệu: ảnh công khai để huấn luyện, ảnh canteen thật để kiểm tra",
    "Phân loại theo quyết định bỏ rác thực tế tại canteen\nCần 2 nguồn dữ liệu: ảnh công khai để huấn luyện và ảnh canteen thật để kiểm tra miền triển khai",
  ],
  ["sh/d0jax03i", "Nguồn dữ liệu & thiết kế nhãn", "Nguồn dữ liệu và thiết kế nhãn"],
  [
    "sh/3ah8rqlg",
    "Bộ chính: Kaggle Recyclable and Household Waste (~15.000 ảnh, 30 lớp) hoặc TrashBusters\nNgoại lai (zero-shot): TACO (crop từ bbox) và RealWaste\nLọc ảnh trùng bằng pHash/MD5; tách sẵn 15% Dev/Holdout ở tuần 1\nHuấn luyện nhãn mịn 10–15 lớp, gộp về 3 nhóm bằng label_map.csv (có đánh số phiên bản)\nThí nghiệm đối chứng: nhãn mịn rồi gộp, so với huấn luyện trực tiếp 3 nhóm",
    "Bộ chính: Kaggle Recyclable and Household Waste khoảng 15.000 ảnh, 30 lớp, hoặc TrashBusters\nTập ngoại lai để kiểm tra zero-shot: TACO crop từ bbox và RealWaste\nLọc ảnh trùng bằng pHash/MD5, tách 15% Dev/Holdout cố định ngay tuần 1\nHuấn luyện nhãn mịn 10-15 lớp, rồi gộp về 3 nhóm bằng label_map.csv có đánh số phiên bản\nĐối chứng RQ1: nhãn mịn rồi gộp so với huấn luyện trực tiếp trên 3 nhóm",
  ],
  ["sh/0ba143al", "Quy ước nhãn & kiểm định chất lượng", "Quy ước nhãn và kiểm định chất lượng"],
  ["sh/i94r6xgz", "ambiguous", "Ảnh mơ hồ"],
  ["sh/jadsz2xk", "Cờ cho ảnh khó, báo cáo riêng", "Gắn cờ ảnh khó và báo cáo riêng"],
  [
    "sh/w72947yt",
    "Viết docs/label_guide.md trước khi chụp; nêu rõ ca ranh giới (ly nhựa dính trà sữa, hộp xốp dính dầu thuộc \"Còn lại\")\nNếu kappa chưa đạt: sửa quy ước và gán lại trước khi thu thập diện rộng\nTập mơ hồ không dùng để tune mô hình",
    "Viết docs/label_guide.md trước khi chụp, nêu rõ ca ranh giới như ly nhựa dính trà sữa hoặc hộp xốp dính dầu thuộc nhóm \"Còn lại\"\nNếu kappa chưa đạt, sửa quy ước và gán lại trước khi thu thập diện rộng\nẢnh mơ hồ không dùng để tinh chỉnh mô hình",
  ],
  ["sh/032tgr6d", "ảnh/nhóm cho Test (≈ 600 ảnh)", "ảnh/nhóm cho Test canteen, khoảng 600 ảnh"],
  ["sh/w32dkbuh", "ảnh/nhóm cho pool few-shot", "ảnh/nhóm cho pool few-shot"],
  ["sh/v2twb6dw", "ảnh/nhóm cho Validation", "ảnh/nhóm cho Validation canteen"],
  [
    "sh/k7mxovud",
    "Camera top-down, nền khay/bàn đồng nhất, 1 vật chính mỗi ảnh; thay đổi có kiểm soát ánh sáng, góc xoay, độ bẩn\nGhi cluster_id = ngày chụp + ca ăn + vật phẩm\nChia Train/Val/Test/Few-shot theo cụm hoặc ngày, không chia ngẫu nhiên từng ảnh để tránh rò rỉ",
    "Chụp top-down trên khay hoặc bàn đồng nhất, một vật chính mỗi ảnh, thay đổi có kiểm soát về ánh sáng, góc xoay và độ bẩn\nGhi cluster_id gồm ngày chụp, ca ăn và vật phẩm cụ thể\nChia Train/Val/Test/Few-shot theo cụm hoặc ngày, không chia ngẫu nhiên từng ảnh để tránh rò rỉ dữ liệu",
  ],
  ["sh/zi98nu94", "Rủi ro đã thống nhất & cách xử lý", "Rủi ro đã thống nhất và cách xử lý"],
  [
    "sh/98rqt4r6",
    "Canteen không cho chụp (rủi ro rất cao): xin phép ngay tuần 1; dự phòng là mua đồ về phòng, dựng góc chụp với khay và bát đĩa chuẩn của trường\nTest canteen quá nhỏ làm khoảng tin cậy rộng: dùng Cluster Bootstrap theo cụm, báo cáo effect size\nKappa chưa đạt: chỉnh label_guide rồi gán lại trước khi chụp diện rộng",
    "Canteen từ chối cho chụp ảnh: xin phép ngay tuần 1, dự phòng bằng cách mua đồ ăn canteen về phòng thực nghiệm và dựng góc chụp mô phỏng\nTest canteen quá nhỏ làm khoảng tin cậy rộng: dùng Cluster Bootstrap theo cluster_id và báo cáo effect size\nKappa chưa đạt: chỉnh label_guide.md rồi gán lại trước khi chụp diện rộng\nNếu trễ ở mốc Go/No-go tuần 5: loại M7 và Canteen-C, dồn thời gian cho Tier B, few-shot và đo trên thiết bị biên",
  ],
  ["sh/dcbud0ra", "Timeline & việc cần làm tiếp", "Tiến độ và việc cần làm tiếp"],
  [
    "sh/zedcfa9g",
    "Tuần 1–2: xin phép canteen, viết label_guide, chụp đợt 1 (100 ảnh), kiểm tra kappa ≥ 0,70\nTuần 3–4: thu thập mở rộng pool và Test canteen\nTuần 5 (Go/No-go): đóng băng Test canteen (≥ 600 ảnh, kappa ≥ 0,7), khóa pool ảnh cốt lõi\nNếu trễ: bỏ nhóm mở rộng M7/Canteen-C, dồn thời gian cho Tier B, few-shot và đo trên thiết bị biên",
    "Tuần 1-2: xin phép canteen, viết label_guide.md, chụp đợt 1 khoảng 100 ảnh và kiểm tra kappa >= 0,70\nTuần 3-4: thu thập mở rộng pool ảnh cốt lõi và Test canteen\nTuần 5 Go/No-go: đóng băng Test canteen >= 600 ảnh, kappa >= 0,70, khóa pool ảnh cốt lõi\nNếu trễ: bỏ nhóm mở rộng M7 và Canteen-C, dồn thời gian cho Tier B, few-shot và đo trên thiết bị biên",
  ],
];

for (const [id, _from, to] of replacements) {
  const target = presentation.resolve(id);
  target.text = to;
}

const candidatePath = path.join(stagingDir, "candidate_dieu_chinh.pptx");
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

const requirements = {
  explicitTotalSlideCount: 7,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
};

await finalizePresentation({
  ...requirements,
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
  receiptPath: path.join(receiptDir, "Team_Meeting_Data_Collection_dieu_chinh.validation.json"),
});

console.log(finalPath);
