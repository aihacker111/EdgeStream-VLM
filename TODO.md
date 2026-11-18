# TODO — EdgeStream-VLM

Thứ tự làm theo milestone. Mỗi milestone kết thúc bằng một thứ **chạy được và đo được**.
Ký hiệu: `[x]` xong · `[ ]` chưa · ⭐ = đóng góp chính của paper.

---

## V. ⭐ Hướng mới: Video Anomaly Detection (UAV là chính, CCTV phụ)

- [x] Ứng viên theo đối tượng: running, falling, lying_still, loitering, close_interaction, crowd_change (ngưỡng theo "kích thước vật", tốc độ đi bộ và dáng người học online từ cảnh)
- [x] Mô hình "bình thường" online: độ hiếm vị trí theo lớp (Poisson, tọa độ đã bù trôi của drone) + kho k-NN đặc trưng chuyển động (tự lọc, lọc tồn tại 3 cửa sổ, khởi động ấm từ video bình thường)
- [x] Tín hiệu pixel: chuyển động dư ngoài mọi box (con vật, vật detector không biết), thay đổi tĩnh (đồ bỏ quên, xả rác)
- [x] Gate VOI: p × severity × độ bất định / chi phí; chỉ hỏi lại khi câu trả lời trước còn mơ hồ
- [x] Chia sẻ xác minh theo đối tượng: một câu trả lời VLM dùng cho mọi giả thuyết trên cùng đối tượng
- [x] Prompt open-set: tiền tố taxonomy cố định (cache được) + 1–3 card theo giả thuyết (theo lớp)
- [x] 24 event card `event_cards/uav_surveillance/` (20 lớp A2Seek + wrong_way, vehicle_on_walkway, illegal_parking, other_anomaly)
- [x] Điểm theo frame: online (nhân quả, tính cả độ trễ VLM) và offline (giao thức chuẩn); trả lời A2Seek đúng định dạng
- [x] Metric: micro/macro AUC, AP, AUC_A, EER (khớp sklearn); loader UCF-Crime, XD-Violence, A2Seek, generic
- [x] `scripts/vad_benchmark.py` (sweep ngân sách, cache perception), `scripts/fit_normal_bank.py`, `python -m edgestream.vad.run`
- [x] Backend `oracle` để tách ảnh hưởng của gate/bộ nhớ khỏi chất lượng VLM
- [x] Test tổng hợp: 3 sự kiện (chạy, ngã, giẫm cỏ) — offline AUC ≈ 0,95, online ≈ 0,77; cảnh bình thường không có sự kiện
- [x] `scripts/run_benchmarks.sh` + `scripts/datasets/fetch.py`: tải A2Seek (+ video của tập test), XD-Violence, UCF-Crime (mirror HF), UIT-ADrone (gdown); chuyển nhãn UIT-ADrone/Drone-Anomaly/MUVAD sang `labels/<video>.npy`; fit bank; sweep ngân sách; chấm A2Seek bằng `evaluate.py` chính thức
- [ ] **Chạy thật** `bash scripts/run_benchmarks.sh all` trên máy; tải Drone-Anomaly thủ công (OneDrive); chờ MUVAD công bố
- [ ] Kiểm tra lại cách ghép nhãn–video của UIT-ADrone sau khi tải (script ghép theo tên, nếu không khớp thì theo thứ tự)
- [ ] Kiểm tra ánh xạ `f_id` → frame video của A2Seek: script tự ước lượng bằng cách so ảnh mẫu với frame video (xem dòng "frame scale check" trong log)
- [ ] Chạy với VLM thật (Ollama `qwen3-vl:4b` / vLLM Qwen3-VL-8B); so sánh với oracle để biết khoảng cách do VLM
- [ ] Hiệu chỉnh `w_unverified`, `w_dense`, `knn_ratio`, `rarity_nats` trên tập val (hiện đặt tay; trên video cao tốc bình thường điểm trung vị ≈ 0,4–0,6 với simulated — cần VLM thật kéo xuống)
- [ ] Học trọng số priority từ nhãn (`PriorityModel.fit`)
- [ ] Bảng chính: AUC/AP theo ngân sách (0, 2, 6, 12 lượt/phút) so với LAVAD, AnomalyRuler, VERA, Holmes-VAU, A2Seek-R1, các phương pháp UAV
- [ ] Ablation: bỏ từng nhánh ứng viên, bỏ VOI, bỏ chia sẻ theo đối tượng, bỏ bù trôi drone, bỏ kho k-NN ấm
- [ ] Đo độ trễ cảnh báo (online) — điểm mạnh so với các phương pháp offline

## M0. Perception (port từ AERO-MOT)

- [x] Port detector + tracker SPEAR/QAM vào `perception/`, API `Perceptor.process(frame) -> FrameOutput`
- [x] Sửa tiền xử lý cho khớp lúc train (RGB + chuẩn hóa ImageNet)
- [x] Đưa dải ứng viên điểm thấp (`cand_floor`) vào tracker (Stage 2.5)
- [x] Chuyển các hằng số riêng của VisDrone (`num_classes == 5`, `class_penalty`, `beta` modulation) ra config
- [x] Vectorize `_build_qam_cost` (cho kết quả giống hệt bản gốc, nhanh hơn ~20 lần ở 300×300; xem `tests/test_qam_cost.py`)
- [x] Load checkpoint strict, tự nhận cấu hình model từ checkpoint, ghim `ultralytics==8.4.215`
- [x] 53 test pass; chạy end-to-end trên video bay thật
- [x] Độ phân giải train của checkpoint MOT: 960×544
- [x] Checkpoint JDE `falcon_yolo_lqcv2_baseline_jde_mot.pth` (epoch 21) chạy được: QAM, ReID gallery, `kappa`, `embedding` đã bật. Trên 12 giây video bay: 3 ID so với 7 ID khi dùng checkpoint chỉ detect
- [ ] (Tùy chọn) Checkpoint đầy đủ có EAD/LQC (`yolo_falcon_lqcv2.yaml`) để khớp hoàn toàn với detector trong paper
- [ ] **Kiểm tra parity với paper:** chạy `perception` trên VisDrone-MOT test-dev, so IDF1/HOTA với `tools/benchmark/visdrone.py` của AERO-MOT (sai lệch phải nhỏ hơn 0,3 điểm)
- [x] Bộ đếm ID riêng cho từng tracker (`IdCounter`); trước đây camera mới khởi động sẽ reset ID của camera đang chạy (`tests/test_id_counter.py`)
- [ ] Kiểm tra zero-shot trên ảnh góc CCTV (UA-DETRAC); fine-tune bằng dữ liệu trộn nếu tụt nhiều
- [ ] Export ONNX/TensorRT (`postprocess.deploy()`), đo trên Jetson (tận dụng `FalconMOT-jetson`). Thêm biến thể scale `n`

## M1. Harness + dữ liệu + đánh giá

- [x] `edgestream/harness/stream.py`: phát lại video theo FPS gốc, đồng hồ logic, nhân quả; ring buffer 30 giây (JPEG)
- [x] `edgestream/harness/budget.py`: kế toán chi phí logic (lượt gọi, visual token, prompt/completion token)
- [x] `edgestream/harness/logger.py`: JSONL cho scene, candidates, gate, vlm_calls, events, route_*
- [x] `edgestream/harness/perception_cache.py`: phát lại `tracks.jsonl` để sweep ngân sách mà không chạy lại detector
- [ ] `data/schema.py`: `Stream`, `EventGT(t_start, t_end, type, track_hint?)`
- [ ] Converter: UIT-ADrone, Drone-Anomaly, MUVAD (UAV); AI City 2026 TAR, ACCIDENT, UCA (mặt đất); tập con OVO-Bench
- [ ] `eval/`: ghép theo temporal IoU ≥ 0,3; F1 theo loại, độ trễ cảnh báo, báo động giả/giờ; chia theo mật độ; vẽ đường cong theo ngân sách
- [ ] Sweep script: lưới B_call × B_mem × B_out chạy trên perception cache

## M2. Baseline

- [ ] B1 `uniform`: gọi VLM đều đặn mỗi k giây, không có bộ nhớ
- [ ] B2 `simplestream`: chỉ N frame gần nhất
- [ ] B3 `fixed_cascade`: ngưỡng cố định, không theo mật độ (đặt `alpha=beta=0` trong config là ra B3)
- [ ] B4 `selectstream` / `folio`
- [ ] B5 perception ablation: YOLO26 + ByteTrack

## M3. Scene state + candidate generator

- [x] `scene/ego_motion.py`: ước lượng chuyển động camera từ nền (LK + RANSAC, che vùng có vật thể); `none` cho CCTV cố định
- [x] `scene/state.py`: lịch sử track, vận tốc so với mặt đất, trường vận tốc + histogram hướng theo ô, mật độ ρ_t và ρ_ref
- [x] `candidates/motion.py`: counter_flow (chỉ ở ô một chiều), stopped_in_flow, sudden_stop, speeding, congestion_forming; ngưỡng tính theo "kích thước vật/giây"
- [x] Mỗi ứng viên mang tín hiệu perception: `quality`, `recovered_frac`, `kappa`
- [ ] `candidates/rules/` (plugin): vùng làn đường, vạch dừng, vỉa hè cho CCTV
- [ ] Hiệu chỉnh ngưỡng trên dữ liệu có nhãn (hiện là giá trị đặt tay)

## M4. ⭐ Gate + budget

- [x] Priority logistic (7 đặc trưng) + `fit()` bằng numpy; trọng số mặc định đặt tay
- [x] Token bucket có dành sẵn phần cho sự kiện nghiêm trọng
- [x] τ_t thích ứng theo mật độ và theo mức bucket; chọn theo p/cost; gửi lại khi p tăng hoặc sau `resend_after_s`
- [x] Test: không vượt ngân sách; τ tăng theo mật độ; phần dành sẵn hoạt động đúng
- [ ] Học trọng số priority từ nhãn (khi có M1 dữ liệu)

## M5. ⭐ Bộ nhớ hai tầng

- [x] Tầng cảnh (thống kê văn bản + keyframe) và tầng thực thể (≤3 crop, quỹ đạo, embedding)
- [x] λ_t thích ứng; loại slot theo e = p·exp(−Δt/T); slot bị loại thành một dòng template
- [x] Re-ID bằng embedding AERO-MOT khi xe xuất hiện lại (chỉ khi chạy perception trực tiếp, cache không có embedding)
- [ ] Pool keyframe thành token thật (hiện chỉ ước lượng số token theo kích thước ảnh)

## M6. VLM

- [x] 5 event card trong `event_cards/traffic/` (có hard negatives)
- [x] Prompt 5 khối cố định thứ tự; Set-of-Mark vẽ sau khi resize để đánh dấu rõ cả xe nhỏ
- [x] Backend `openai` (vLLM/SGLang/llama.cpp/LM Studio/Ollama): JSON schema + logprobs, tự hạ cấp nếu server không hỗ trợ
- [x] Backend `transformers` (Qwen3-VL), backend `simulated` cho test hệ thống
- [x] Parser JSON chịu lỗi; độ tin cậy từ logprob
- [ ] **Chạy với VLM thật** (Ollama `qwen3-vl:4b` trên Mac, hoặc vLLM trên GPU) — chưa làm được trong môi trường của tôi vì không tải được trọng số
- [ ] Đo prefix caching (vLLM `--enable-prefix-caching`)
- [x] Event card cho UAV và giám sát chung (`event_cards/uav_surveillance/`)

## M7. Gộp sự kiện + đầu ra

- [x] Gộp theo loại + track trong `merge_gap_s`, cooldown theo (loại, ô)
- [x] Clip bằng chứng + keyframe từ ring buffer; định tuyến command_center / review_queue / analytics
- [x] Video `pipeline.mp4` có HUD (mật độ, τ, bucket, bộ nhớ, số lượt gọi)
- [ ] Làm mờ mặt/biển số (EgoBlur) trước khi lưu bằng chứng
- [ ] Luồng câu hỏi vận hành (query path)

## M8. Thí nghiệm

- [ ] E1 perception (mAP, HOTA, IDF1) trên VisDrone/UAVDT + domain CCTV
- [ ] E2 ⭐ đường cong accuracy theo ngân sách (lưới B_call × B_mem × B_out × cỡ VLM)
- [ ] E3 ⭐ stress test mật độ, gộp mọi domain
- [ ] E4 tổng quát hóa trên các benchmark công khai
- [ ] E5 ablation: cổng mật độ, tầng thực thể, tín hiệu perception, ép JSON, hard negatives
- [ ] E6 ảo giác/nhất quán (CCTVBench)
- [ ] E7 độ trễ thật (Jetson / điện thoại)
- [ ] E8 chất lượng tracking ảnh hưởng thế nào đến F1 sự kiện và báo động giả (AERO-MOT so với ByteTrack)

## M9. Viết paper

- [ ] Hình kiến trúc, hình đường cong theo ngân sách, hình F1 theo mật độ
- [ ] Mục Limitations: chưa đo nhiệt độ và độ bền khi chạy dài ngoài trời
- [ ] Phần đạo đức: làm mờ dữ liệu, con người vẫn là người quyết định cuối cùng

---

### Việc vặt
- [ ] `git init`, CI chạy `pytest`, pre-commit (ruff)
- [ ] Cố định seed; lưu config đã resolve cùng mỗi lần chạy
- [ ] Ghi chú giấy phép: `perception/` là AGPL-3.0 nên toàn repo nên dùng AGPL
