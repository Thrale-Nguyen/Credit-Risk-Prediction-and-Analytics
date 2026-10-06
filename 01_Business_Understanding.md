# ĐỀ TÀI PRJ-01 · PHÂN TÍCH VÀ DỰ BÁO RỦI RO VỠ NỢ TÍN DỤNG CÁ NHÂN

## GIAI ĐOẠN 1 — BUSINESS UNDERSTANDING (Hiểu bài toán & Xác lập mục tiêu kinh doanh)

|                                     |                                                                                                                                 |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **Chuyên ngành**            | Tài chính – Ngân hàng                                                                                                      |
| **Bộ dữ liệu**             | Home Credit Default Risk (Kaggle, 2017) — phương án dự phòng: Lending Club Loan Data                                      |
| **Grain dữ liệu**           | 1 dòng = 1 hồ sơ vay (`SK_ID_CURR`), 307 511 hồ sơ, liên kết 1:n với 5 bảng phụ (~31,2 triệu dòng)                |
| **Bài toán ML**             | Phân loại nhị phân có giám sát (supervised binary classification) — xuất ra**xác suất vỡ nợ (PD)**           |
| **Mô hình trọng tâm**     | Logistic Regression (baseline giải trình) · Random Forest Classifier (bất tuyến tính)                                     |
| **Kỹ thuật đặc trưng**   | `pd.merge()` đa cấp · `groupby().agg()` trên bảng lịch sử · xử lý mất cân bằng lớp                           |
| **Ngày lập / phiên bản**  | 05/10/2026 · v1.0                                                                                                              |
| **Vị trí trong quy trình** | Giai đoạn 1/6 theo CRISP-DM — tài liệu này**khóa phạm vi, chỉ tiêu và thiết kế biến** cho 5 giai đoạn sau |

### Quy chế dẫn chứng trong tài liệu

Vì kỷ luật nghiên cứu yêu cầu tách bạch "số đã kiểm chứng" khỏi "số suy diễn", mọi con số trong tài liệu mang một trong ba nhãn:

| Nhãn         | Ý nghĩa                                                                                                                                                                 | Cách kiểm chứng                           |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| **[K]** | **Kiểm chứng** — trích từ tài liệu chính thức của bộ dữ liệu / văn bản công khai của NHNN & báo chí kinh tế, đã đối chiếu chéo            | Dẫn nguồn ở Phụ lục C                   |
| **[M]** | **Mô hình** — hệ quả giải tích của mô hình kinh tế ở mục 1.5 với tham số nêu tường minh; **không phải** kết quả đo trên dữ liệu vay | Chạy lại`tools/figures_and_economics.py` |
| **[C]** | **Chờ kiểm chứng** — giả định làm việc, sẽ được đo trực tiếp trên file CSV ở Giai đoạn 2 bằng `tools/verify_dataset_facts.py`                | Xem mục 7                                   |

> **Cảnh báo liêm chính học thuật:** mọi con số hiệu năng mô hình (AUC, KS…) trong tài liệu này đều là **chỉ tiêu phải đạt**, không phải kết quả đã đạt. Giai đoạn 1 chưa chạy một mô hình nào trên dữ liệu.

---

## 0. TÓM TẮT ĐIỀU HÀNH

**Vấn đề.** Cho vay tiêu dùng cá nhân là nghiệp vụ có tỷ lệ lỗ kỳ vọng cao nhất và tốc độ tăng trưởng nhanh nhất trong danh mục ngân hàng bán lẻ. Một ngân hàng/ công ty tài chính không thể biết trước ai sẽ không trả nợ, nhưng bắt buộc phải quyết định trong vài giờ. Kết quả là chi phí của **thông tin bất đối xứng** — người vay biết mình sẽ trả hay không trước cả ngân hàng — được trả bằng nợ xấu, bằng chi phí dự phòng và bằng việc từ chối oan khách hàng tốt.

**Dữ liệu.** Bộ Home Credit cung cấp đúng thứ mà một mô hình xếp hạng tín dụng hiện đại cần: ngoài 121 biến khai báo trên hồ sơ vay **[K: 122 cột, trong đó 1 cột nhãn]**, còn có *lịch sử quan hệ tín dụng tại tổ chức khác* (bureau), *toàn bộ các đơn xin vay trước đây kể cả đơn bị từ chối* (previous_application), và *hành vi trả nợ từng kỳ hạn/tháng* (installments, POS, credit card). Tổng cộng 31,19 triệu dòng **[K]**, trung bình 1 hồ sơ gánh ≈100 dòng dữ liệu ở phía sau **[K: suy ra từ tỉ lệ số dòng]**. Đây là lý do kỹ thuật `groupby().agg()` rồi mới `pd.merge()` không phải "mẹo lập trình" mà là **điều kiện phương pháp luận** để không phá vỡ grain của mẫu (xem Hình 1).

**Giá trị.** Mô hình xếp hạng chỉ có giá trị ở chỗ nó *tách được* nhóm xấu ra khỏi nhóm tốt. Với cấu trúc chi phí của một khoản vay tiêu dùng vô thế điển hình (biên lợi nhuận ròng trước rủi ro 15%, LGD 60%), xác suất vỡ nợ hoà vốn của một hồ sơ là **PD\* = 20%**. Mọi hồ sơ có PD > 20% cần bị từ chối; mọi cải thiện về năng lực xếp hạng dịch trực tiếp thành lợi nhuận và thành nợ xấu tránh được. Mô hình **[M]** ở mục 1.5 cho thấy: nâng ROC-AUC từ 0,72 lên 0,78 giúp tăng lợi nhuận danh mục ~0,43 điểm % quy mô vay và hạ nợ xấu của sổ được duyệt từ 7,0% xuống 5,7% — với danh mục giả định 100 000 hồ sơ/năm × 80 tỷ đồng thì tương đương ~59 tỷ đồng lợi nhuận gia tăng và 2,31 tỷ đồng dư nợ xấu tránh được mỗi năm **[M]**.

**Bài toán được chọn.** Phân loại nhị phân, nhãn `TARGET` (1 = không trả được nợ), 8,07% dương tính — mất cân bằng 1 : 11,4 **[K]**. Deliverable không phải "nhãn 0/1" mà là **thang điểm PD được hiệu chuẩn** (scorecard) + **ngưỡng duyệt tối ưu theo chi phí**.

**Chỉ tiêu cam kết (rút gọn).** ROC-AUC ≥ 0,78 · KS ≥ 0,42 · Lift decile-1 ≥ 3,5 · PR-AUC ≥ 0,30 · recall ở 10% điểm thấp nhất ≥ 0,35 · pipeline 31,2 triệu dòng → 1 ma trận đặc trưng trong ≤ 45 phút / ≤ 16 GB RAM · hoàn tất trước **28/12/2026**.

---

## 1. BỐI CẢNH KINH TẾ & NỖI ĐAU DOANH NGHIỆP

### 1.1. Bối cảnh vĩ mô: tăng trưởng tín dụng nhanh đặt trên nền nợ xấu cao

Ba con số định hình tính thời sự của đề tài tại Việt Nam **[K]**:

1. **Tín dụng tăng trưởng rất nhanh.** Tăng trưởng tín dụng toàn hệ thống năm 2023 là 13,78%, năm 2024 là 15,09%; đến 30/9/2025 đạt +13,86% so với đầu năm (+20,22% so với cùng kỳ). Theo NHNN, tính đến 24/12/2025 tổng dư nợ tín dụng nền kinh tế **vượt 18,4 triệu tỷ đồng, tăng 17,87%** so với cuối 2024 — tức cao hơn hẳn mục tiêu 16% đặt ra đầu năm.
2. **Chất lượng tài sản không theo kịp.** Tỷ lệ nợ xấu nội bảng toàn ngành ở mức **4,55% (cuối 2023) → 4,75% (7/2024) → 4,3% (1/2025)** theo báo cáo tổng kết thi hành Nghị quyết 42 của NHNN; riêng 27 ngân hàng niêm yết, tổng nợ xấu quý I/2025 vượt 266 nghìn tỷ đồng (+18,5% so với cùng kỳ), trong đó nợ nhóm 3 (dưới chuẩn) tăng hơn 37%.
3. **Nghịch lý về con số.** Cũng số liệu chính phủ, nợ xấu nội bảng cuối tháng 10/2025 được nêu ở mức **1,64%** — nhưng *không bao gồm* nợ xấu của 5 nhà băng mua lại bắt buộc/kiểm soát đặc biệt. Hai con số không mâu thuẫn: chúng khác **phạm vi thống kê**. Đây chính là loại bẫy "tỷ lệ phụ thuộc mẫu số" mà Giai đoạn 2 của đề tài phải nhận diện khi định nghĩa tỷ lệ vỡ nợ.

**Suy luận cho nghiệp vụ.** Tín dụng bán lẻ tăng nhanh nhất ở đúng khúc rủi ro nhất (vay tiêu dùng, vay tiền mặt, vay qua POS không tài sản đảm bảo), trong khi khả năng đo lường rủi ro của nhiều định chế vẫn dựa vào phán đoán chuyên gia và các bộ chỉ tiêu cứng. Mỗi chu kỳ nới tăng trưởng đều để lại một thế hệ nợ xấu mới. Vì vậy giá trị của đề tài không nằm ở việc "dự đoán giỏi hơn" một cách trừu tượng, mà ở việc **cắt được độ trễ giữa tốc độ tăng trưởng và năng lực lượng hóa rủi ro**.

### 1.2. Vì sao đây là vấn đề kinh tế, không chỉ là vấn đề thống kê

Ba khiếm khuyết thị trường đứng sau mọi con số ở trên:

- **Lựa chọn đối nghịch (adverse selection).** Khách hàng biết rủi ro của mình tốt hơn ngân hàng. Nếu ngân hàng định giá chung một lãi suất cho mọi người, lãi suất đó cao với người tốt và rẻ với người xấu → danh mục tự xấu đi. Mô hình PD cho phép *tách giá theo rủi ro* thay vì tách theo cảm nhận.
- **Rủi ro đạo đức (moral hazard).** Sau giải ngân, hành vi trả nợ thay đổi. Ý nghĩa với đề tài: **biến hành vi** (DPD, tỷ lệ trả đủ kỳ hạn, số lần gia hạn) luôn là nhóm biến quyền lực nhất — đây là giả thuyết H2 ở mục 3.
- **Chi phí xử lý thông tin.** Với quy mô 100 000 hồ sơ/năm, chi phí thẩm định thủ công ~0,2% giá trị khoản vay **[M]** đã là 160 tỷ đồng/năm chỉ để "xét". Tự động hóa không chỉ giảm nợ xấu; nó giảm cả chi phí ra quyết định — đây là lý do nghiệp vụ gọi mô hình này là *application scoring*.

**Điểm mấu chốt về mặt kinh tế học:** trong cho vay tiêu dùng, **sai lầm đắt nhất không phải là duyệt nhầm một khách xấu, mà là không biết mình đang phạm bao nhiêu sai lầm**. Một ngân hàng không có mô hình không có *đường cong ROC* — nên không biết phải đổi bao nhiêu doanh thu để lấy một điểm nợ xấu. Giai đoạn 1 của đề tài, vì thế, đặt trọng tâm vào việc **xây hàm chi phí** (mục 1.5) chứ không chỉ vào việc chọn thuật toán.

### 1.3. Bảng nỗi đau doanh nghiệp (Business Pain Points)

| #  | Nỗi đau                                                                           | Cơ chế gây tổn thất                                                                                                                                                                                                                        | Đại lượng đo được trong đề tài                                                                | Mức độ                       |
| -- | ----------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | ------------------------------- |
| P1 | **Nợ xấu và tổn thất tín dụng**                                        | Vỡ nợ → mất gốc (LGD 55–75% với vay tiêu dùng không tài sản đảm bảo) → ăn vào lợi nhuận và vốn chủ                                                                                                                       | Tỷ lệ vỡ nợ danh mục (8,07% trên mẫu**[K]**), tổng tổn thất kỳ vọng                          | Nghiêm trọng                  |
| P2 | **Chi phí dự phòng leo thang**                                             | Mỗi đồng nợ xấu buộc trích lập DPRR theo nhóm nợ → giảm lợi nhuận sau thuế ngay cả khi chưa xử lý xong tài sản; đồng thời siết CAR                                                                                     | Tỷ lệ vỡ nợ theo nhóm khách hàng; chất lượng dự báo PD (calibration)                         | Nghiêm trọng                  |
| P3 | **Quyết định thủ công, chậm và không nhất quán**                    | Chuyên viên xét duyệt phụ thuộc kinh nghiệm cá nhân → cùng một hồ sơ, hai kết luận khác nhau; thời gian phản hồi làm mất khách hàng vào tay đối thủ/Fintech                                                         | Tỷ lệ duyệt, độ ổn định của ngưỡng; năng lực xử lý tự động (hồ sơ/giờ)              | Trung bình – cao              |
| P4 | **Từ chối oan khách hàng tốt (false positive)**                          | Mất doanh thu margin + mất vòng đời khách hàng; mặt khác *hồ sơ bị từ chối không có nhãn* → mô hình luyện trên người được duyệt bị **chệch chọn mẫu**, đánh giá thấp rủi ro thực của quần thể | `previous_application` có cả đơn bị từ chối → **reject inference** (RQ6)                 | Trung bình, hay bị bỏ qua    |
| P5 | **Dữ liệu nằm rời rạc, không khai thác được**                       | Thông tin nằm ở 5–6 bảng 1:n; 31,2 triệu dòng chưa được bào phẳng → đặc trưng mạnh nhất (hành vi trả nợ) không vào được mô hình                                                                                    | Số đặc trưng sinh ra/bảng; incremental AUC theo bảng (RQ2)                                         | Cao (nút thắt kỹ thuật)     |
| P6 | **Mất cân bằng lớp làm lệch mô hình**                                 | 91,93% hồ sơ nhãn 0 → mô hình "dự đoán tất cả = 0" đạt**accuracy 91,93%** nhưng vô dụng (AUC 0,5). Nếu dùng accuracy/loss chuẩn, mô hình học cách… không phát hiện gì                                         | PR-AUC, recall ở decile rủi ro cao, cân bằng lớp (RQ4)                                              | Cao (bẫy phương pháp luận) |
| P7 | **Không giải trình được với hội đồng rủi ro & cơ quan thanh tra** | Mô hình hộp đen không đáp ứng yêu cầu thông báo lý do từ chối cho khách hàng và không qua được thẩm định mô hình nội bộ; dùng biến giới tính/khu vực còn gây rủi ro phân biệt đối xử                  | WOE/IV, hệ số LogReg, SHAP, kiểm định công bằng theo nhóm (G6)                                   | Trung bình – cao              |
| P8 | **Ngưỡng duyệt đặt theo quán tính, không theo kinh tế**              | Scorecut-line 0,5 hoặc "cảm tính" → hoặc thắt quá mất doanh thu, hoặc lỏng quá vỡ nợ; không ai tối ưu hàm lợi nhuận                                                                                                          | Ngưỡng tối ưu PD\*, ma trận chi phí sai lầm, đường cong lợi nhuận–tỷ lệ duyệt (mục 1.5) | Cao, ít được nhận diện    |

**Tổng hợp thành một câu:** doanh nghiệp không thiếu dữ liệu và cũng không thiếu mô hình; họ thiếu **(i)** đặc trưng hành vi được kết nối đúng cách, **(ii)** một thước đo hiệu năng chịu ảnh hưởng của mất cân bằng, và **(iii)** một ngưỡng quyết định được suy ra từ chi phí. Đề tài giải quyết đúng ba khoảng trống đó.

### 1.4. Ý nghĩa thực tiễn ba tầng

| Tầng         | Người thụ hưởng                    | Kết quả kỳ vọng                                                                                                                                                      |
| ------------- | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Vận hành    | Khối QLRRTD / tác nghiệp phê duyệt | Tự động hoá vòng 1: hồ sơ "xanh" duyệt trong giây, hồ sơ "đỏ" chặn sớm, chuyên viên chỉ xử lý vùng xám → giảm thời gian & khối lượng xử lý |
| Chiến lược | Khối danh mục & ALCO                  | Scorecut-line, hạn mức và lãi suất theo phân khúc PD; tối ưu biên lợi nhuận điều chỉnh theo rủi ro (risk-adjusted margin)                                |
| Vĩ mô       | Người vay & cơ quan quản lý        | Mở rộng tài chính toàn diện có kiểm soát (khách hàng "thin file" không còn bị loại vì thiếu lịch sử); giảm nợ xấu hệ thống                       |

### 1.5. Định lượng giá trị kinh doanh: từ AUC sang đồng tiền

Đây là phần biến một bài tập ML thành một đề tài tài chính. Toàn bộ là **[M]** — mô hình giải tích một kỳ, tham số nêu tường minh, chạy tại `tools/figures_and_economics.py`.

**Quy ước.** Chuẩn hoá quy mô mỗi khoản vay = 1 đơn vị dư nợ (EAD = 1).

| Tham số                                           | Ký hiệu | Giá trị dùng  | Căn cứ chọn                                                                                |
| -------------------------------------------------- | --------- | ---------------- | --------------------------------------------------------------------------------------------- |
| Biên lợi nhuận ròng trước rủi ro tín dụng | `r`     | 15% / kỳ vay    | Lãi cho vay tiêu dùng 20–30%/năm trừ chi phí vốn ~6–7% và chi phí hoạt động ~5% |
| Tổn thất khi vỡ nợ                             | `LGD`   | 60%              | Vay tiền mặt/POS không tài sản đảm bảo; thu hồi qua xử lý nợ thấp                |
| Tỷ lệ vỡ nợ trước của quần thể            | `π`    | 8,07%            | Nhãn`TARGET`, mẫu 307 511 **[K]**                                                   |
| Chi phí xử lý một hồ sơ                      | `c`     | 0,2% quy mô vay | Chi phí tác nghiệp + CIC query                                                             |

**Xác suất vỡ nợ hoà vốn.** Duyệt một hồ sơ có PD = *p* có kỳ vọng dương khi

```
(1 − p)·r − p·LGD > 0    ⟺    p < p* = r / (r + LGD) = 15% / 75% = 20%
```

`p* = 20%` **chính là ngưỡng duyệt tối ưu về kinh tế**, và cũng chính là điểm cắt trong ma trận chi phí sai lầm: chi phí từ chối oan một khách tốt = `r` (mất margin), chi phí duyệt nhầm một khách xấu = `LGD` (mất vốn); `p* = c_FP/(c_FP + c_FN)`. Không cần mô hình nào để biết công thức này — **mô hình chỉ có nhiệm vụ ước lượng p**.

**Kết quả mô phỏng 400 000 hồ sơ** (điểm tín dụng phân phối chuẩn, tách theo AUC; mỗi AUC tương ứng một năng lực xếp hạng):

| Năng lực mô hình                              | Tỷ lệ duyệt tối ưu | Nợ xấu của sổ được duyệt | Lợi nhuận / 100 hồ sơ | So với "duyệt tất cả" | % hồ sơ vỡ nợ bị chặn đúng |
| ------------------------------------------------- | ----------------------- | -------------------------------- | ------------------------- | ------------------------- | ---------------------------------- |
| AUC = 0,50 (ngẫu nhiên)                         | 100%                    | 8,03%                            | 8,78                      | —                        | 0%                                 |
| AUC = 0,68                                        | 95,9%                   | 7,34%                            | 8,91                      | +0,15                     | 12,6%                              |
| **AUC = 0,72** (LogReg khả thi)            | 95,3%                   | 7,00%                            | 9,09                      | +0,31                     | 16,8%                              |
| **AUC = 0,78** (RF + đặc trưng hành vi) | 90,9%                   | 5,71%                            | 9,56                      | **+0,74**           | **34,9%**                    |
| AUC = 0,81 (trần thực tế của bộ dữ liệu)   | 90,5%                   | 5,30%                            | 9,80                      | +1,04                     | 40,4%                              |

![Kinh tế học của ngưỡng duyệt](assets/fig2_economics.png)

**Ba kết luận rút ra cho việc đặt mục tiêu:**

1. **"Duyệt tất cả" vẫn có lãi (8,75%)** — vì π = 8,07% < p\* = 20%. Điều này giải thích hành vi thực tế của các công ty tài chính tiêu dùng: họ sống bằng *volume* và chấp nhận nợ xấu cao. Nhưng nó cũng có nghĩa **mọi điểm AUC tăng thêm đều là tiền**, chứ không phải là "độ chính xác trang trí".
2. **Giá trị của 6 điểm AUC (0,72 → 0,78) lớn hơn nhiều giá trị của 6 điểm tiếp theo** trong vùng π < p\*, nhưng quan trọng hơn: **đường cong lợi nhuận đạt đỉnh rồi giảm** (xem Hình 2, điểm cực đại ~91% duyệt). Nghĩa là *không* có "càng chặt càng tốt" — và đó là lý do P8 là nỗi đau thật.
3. **Ngưỡng suy từ chi phí gần như trùng ngưỡng tối ưu:** chính sách cơ học "duyệt nếu PD̂ < 20%" cho lợi nhuận 9,087/100 hồ sơ, chỉ thấp hơn cực đại 0,007 điểm. ⇒ **Đề tài không cần thuật toán tìm scorecut-line phức tạp**; chỉ cần một PD được hiệu chuẩn tốt. Hệ quả phương pháp luận: *calibration* phải là chỉ tiêu bắt buộc, không phải phụ lục.

**Quy đổi tiền tệ** cho danh mục giả định 100 000 hồ sơ/năm × 80 tỷ đồng/hồ sơ (= 8 000 tỷ dư nợ): nâng AUC từ 0,72 lên 0,78 ≈ **+59,2 tỷ đồng lợi nhuận/năm**, dư nợ xấu giảm từ 6,46 tỷ xuống 4,15 tỷ, và 34,9% khách hàng vỡ nợ bị chặn ngay ở cửa. **[M]**

### 1.6. Ràng buộc pháp lý & quản trị mô hình (nguồn lực bất định của bài toán)

| Khung                                                                                                                  | Nội dung ảnh hưởng trực tiếp đến đề tài                                              | Hệ quả cho thiết kế                                                                                                                             |
| ---------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Basel II/III — cách tiếp cận IRB                                                                                   | Ngân hàng tự xây mô hình nội bộ để ước lượng PD/LGD/EAD, phục vụ tính vốn     | Bài toán phải được phát biểu thành**PD**, không phải "nhãn 0/1"; đòi hỏi hiệu chuẩn và ổn định theo thời gian           |
| Phân loại nợ & trích lập DPRR (TT 11/2021/TT-NHNN và các văn bản sửa đổi/gia hạn như TT 02/2023/TT-NHNN) | Nhóm nợ 1→5 theo số ngày quá hạn; nhóm 5 (>90 ngày) là "nợ có khả năng mất vốn" | Định nghĩa "vỡ nợ" trong tài liệu được**neo vào ngưỡng 90 ngày** để đối chiếu được với số liệu quản trị (mục 4.2) |
| IFRS 9 / dự phòng tổn thất kỳ vọng (ECL)                                                                         | Ba giai đoạn, PD qua toàn bộ vòng đời, yếu tố vĩ mô                                  | Nhấn mạnh*calibration* và *forward-looking*; giải thích vì sao accuracy không đủ                                                       |
| Luật Các TCTD 2024 & lộ trình pháp lý thay Nghị quyết 42 (hết hiệu lực 31/12/2023)                          | Siết quản trị, tăng trách nhiệm giải trình của hội đồng thành viên                | Tính giải trình (SHAP/WOE) là yêu cầu, không phải điểm cộng                                                                              |
| Nghị định 13/2023/NĐ-CP về bảo vệ dữ liệu cá nhân                                                           | Dữ liệu nhân thân, tài chính là dữ liệu cá nhân cần có cơ sở xử lý             | Loại biến nhạy cảm (giới, dân tộc…) khỏi mô hình; kiểm định công bằng (G6)                                                          |

> **Lưu ý cho báo cáo:** các số hiệu thông tư cần được đối chiếu bản hợp nhất có hiệu lực tại thời điểm bảo vệ. Đề tài dùng chúng như **neo khái niệm** (định nghĩa vỡ nợ, yêu cầu giải trình), không phải như đối tượng phân tích pháp lý.

### 1.7. Phát biểu bài toán kinh doanh → bài toán dữ liệu

**Phát biểu kinh doanh.** *Trong danh mục cho vay tiêu dùng, làm sao duyệt được nhiều hồ sơ nhất có thể mà vẫn giữ nợ xấu của sổ được duyệt thấp hơn ngưỡng mà biên lợi nhuận chịu đựng được?*

Ba bài toán con nghiệp vụ, và phạm vi đề tài:

| Bài toán con                                       | Câu hỏi nghiệp vụ                                | Đề tài                                                                                                               |
| ---------------------------------------------------- | ---------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **A. Application scoring** — duyệt/từ chối | Khách này có nên cho vay không?                 | ✅**Trọng tâm**                                                                                                 |
| **B. Limit & pricing**                         | Cho vay bao nhiêu, lãi suất nào?                 | ⚠️ Một phần: phân tích theo phân khúc PD & DTI, không tối ưu hạn mức                                       |
| **C. Behavioural scoring / Early warning**     | Khách hiện hữu sẽ trễ hạn trong 3 tháng tới? | ❌ Ngoài phạm vi (cần dữ liệu khoản vay đang chạy; bộ dữ liệu này là hồ sơ*tại thời điểm xin vay*) |

**Dịch sang ngôn ngữ dữ liệu (CRISP-DM bước chuyển giao):** cho 307 511 hồ sơ vay với 121 biến khai báo và ≈30,8 triệu dòng dữ liệu lịch sử liên kết, xây một **hàm xếp hạng** `f: X → [0,1]` ước lượng xác suất vỡ nợ trong cửa sổ 12 tháng, sao cho (i) khả năng phân tách đạt AUC ≥ 0,78, (ii) xác suất được hiệu chuẩn để cắt ở p\* = 20%, (iii) top-20% rủi ro nhất bắt được ≥ 35% số vụ vỡ nợ, (iv) mô hình giải trình được bằng hệ số & SHAP.

### 1.8. Lựa chọn dữ liệu: vì sao Home Credit thay vì Lending Club

| Tiêu chí                                                             | Home Credit Default Risk ✅                                                         | Lending Club ⚠️                                                                                                          |
| ---------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Khớp kỹ thuật trọng tâm của đề tài (merge đa cấp + groupby) | 1 bảng chính + 5–6 bảng phụ 1:n,**bắt buộc** dùng `groupby().agg()` | 1 bảng phẳng 2,2 triệu dòng → kỹ thuật đa bảng trở thành hình thức                                            |
| Nhãn                                                                  | `TARGET` nhị phân sạch, mất cân bằng 1:11,4                                 | `loan_status` đa trạng thái → phải tự mã hóa lại (Charged Off/Default vs Fully Paid), phụ thuộc định nghĩa |
| Tính nhất quán thời gian                                           | Một đợt cấp thư 2016 → ít nhiễu chu kỳ                                     | 2007–2018, gồm khủng hoảng tài chính → hiệu năng mô hình bị lẫn với diễn biến vĩ mô                      |
| Tính khả dụng                                                       | Còn tải trên Kaggle, có data dictionary 122 cột                                | Nguồn gốc (Upstart) đã dừng cấp mới; dữ liệu phân mảnh                                                          |
| Bối cảnh địa lý                                                   | Thị trường mới nổi, cho vay tiêu dùng —**gần Việt Nam hơn**        | Thị trường Mỹ, lãi suất & pháp lý khác biệt                                                                      |

**Vì sao vẫn nêu Lending Club:** nếu tài nguyên tính toán không cho phép xử lý 31,2 triệu dòng, Lending Club là **phương án dự phòng** một bảng (2,2 triệu dòng) để giữ nguyên toàn bộ khung phương pháp. Khi đó các mục RQ2, G3 và G5 phải được viết lại. Đây là kế hoạch B được khai báo ngay từ đầu để tránh "chữa cháy" giữa kỳ.

### 1.9. Phạm vi và giả định

**Trong phạm vi:** dữ liệu 2016 của Home Credit (Cộng hòa Séc); hồ sơ vay tiêu dùng; dự báo vỡ nợ trong 12 tháng; hai họ mô hình LogReg & RF; kỹ thuật xử lý mất cân bằng; phân tích chi phí–lợi ích ở ngưỡng duyệt.

**Ngoài phạm vi:** LGD/EAD (không có dữ liệu thu hồi nợ), chính sách hạn mức động, mô hình theo vòng đời khách hàng, tối ưu danh mục cấp vốn, deep learning/gradient boosting nâng cao (chỉ dùng làm **tham chiếu trần hiệu năng** nếu thời gian cho phép, không phải mô hình chính), và mọi suy nhân quả ("biến X *gây ra* vỡ nợ").

**Giả định làm việc:** (A1) `TARGET` phản ánh đúng vỡ nợ theo định nghĩa nội bộ của Home Credit; (A2) quan hệ hành vi–vỡ nợ năm 2016 đủ ổn định để minh hoạ phương pháp, **không** đủ để triển khai cho một ngân hàng Việt Nam năm 2026; (A3) `AMT_*` tính bằng CZK; (A4) người vay trong bảng đã được duyệt nên mẫu **không** đại diện cho toàn bộ quần thể xin vay → mọi ước lượng rủi ro là cận dưới (see RQ6).

---

## 2. MỤC TIÊU NGHIÊN CỨU THEO NGUYÊN TẮC SMART

Nguyên tắc đặt chỉ tiêu: mỗi mục tiêu phải có **baseline** (nếu không có baseline thì phải *tạo* baseline làm chuẩn so sánh), **thước đo duy nhất**, **ngưỡng đạt / ngưỡng xuất sắc**, và **ngày kiểm tra**. Mọi chỉ tiêu hiệu năng được đo trên tập kiểm định theo nhóm (GroupKFold trên `SK_ID_CURR`), không đo trên tập huấn luyện.

| Mã          | Mục tiêu (Specific)                                                                                                                    | Thước đo (Measurable)                                                                                                                                                                       | Baseline                                                    | Ngưỡng ĐẠT                                                                                                                                                                            | Ngưỡng XUẤT SẮC                                                                                               | Khả thi (Achievable)                                                                                                                                          | Gắn nỗi đau | Hạn chót (Time-bound)                                         |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- | --------------------------------------------------------------- |
| **G1** | Xây dựng mô hình phân loại nhị phân dự báo vỡ nợ có**năng lực phân tách** đủ dùng cho quyết định duyệt vay | ROC-AUC; KS; Gini = 2·AUC−1                                                                                                                                                                  | Dự đoán tất cả = 0:**AUC 0,50**, accuracy 91,93% | AUC ≥**0,76** · KS ≥ 0,40                                                                                                                                                        | AUC ≥**0,78** · KS ≥ 0,42                                                                                | LogReg 1 bảng ≈0,72 và RF/GBoost với đặc trưng hành vi ≈0,78–0,80 là mặt bằng đã công bố rộng rãi của bộ dữ liệu**[K, cần tái lập]** | P1, P3         | 14/12/2026 (hết W10)                                           |
| **G2** | Tối ưu chỉ tiêu**trong vùng mất cân bằng**, nơi accuracy vô nghĩa                                                       | PR-AUC (AP); Lift decile-1; Recall@10% điểm thấp nhất; F0.5                                                                                                                                | Dummy: AP = π = 0,0807; Recall@10% = 0,10                  | AP ≥**0,28** · Lift₁ ≥ 3,0 · Recall@10% ≥ 0,32                                                                                                                                | AP ≥**0,30** · Lift₁ ≥ 3,5 · Recall@10% ≥ 0,35                                                        | Ở AUC 0,78, cấu trúc xếp hạng cho ~35% vỡ nợ nằm trong 20% dưới cùng**[M]**                                                                         | P6, P1         | 14/12/2026                                                      |
| **G3** | Chứng minh**giá trị gia tăng của dữ liệu lịch sử 5 bảng phụ** — chứ không chỉ "nhiều biến hơn"                   | ΔAUC, ΔKS, ΔGini của mỗi khối đặc trưng qua**thí nghiệm loại bỏ (ablation)** 5 cấu hình                                                                                   | Hồ sơ vay đơn bảng (A₀)                               | ΔAUC ≥**+0,02** khi thêm toàn bộ 5 bảng; ít nhất 1 bảng cho ΔAUC > 0                                                                                                      | ΔAUC ≥ +0,025 và bảng*installments_payments* nằm trong top 2 về đóng góp                               | Chính kỹ thuật`groupby().agg()` sinh biến tổng hợp; tài liệu về bộ dữ liệu cho thấy biến hành vi là nhóm mạnh nhất **[K]**          | P5, P1         | 02/11/2026 (hết W4) cho bảng phân tích; xác nhận tại W10 |
| **G4** | Xử lý**mất cân bằng dữ liệu** một cách có kiểm soát, không phá vỡ ý nghĩa xác suất                              | So 5 phương án: (a) không xử lý, (b)`class_weight`, (c) NearMiss/undersample, (d) SMOTE, (e) chỉ dịch ngưỡng; đo AP, Recall@10%, **Brier**, độ dốc đường hiệu chuẩn | (a) không xử lý: AUC 0,76 nhưng PD̂ thấp hệ thống   | Tìm được 1 phương án cải thiện AP ≥ +0,02**và** giữ Brier ≤ 0,07, calibration slope ∈ [0,9; 1,1]                                                                      | Cải thiện AP mà**không** hy sinh calibration; nếu SMOTE làm hỏng hiệu chuẩn, chứng minh bằng số | 8,07% dương tính là vùng SMOTE có thể gây nhiễu nhưng class_weight/threshold-moving hầu như luôn an toàn                                         | P6, P2         | 21/12/2026 (hết W11)                                           |
| **G5** | Xây**pipeline tái lập được** trên 31,2 triệu dòng, đáp ứng ràng buộc phần cứng của người thực hiện            | Thời gian chạy end-to-end; RAM đỉnh; số dòng đầu ra của mỗi bước;`assert` grain = 307 511                                                                                        | Load thô 8 file CSV > 30 GB RAM, > 40 phút                | ≤**45 phút**, ≤ **16 GB** RAM, 0 lỗi grain, 100% tái lập khi chạy lại                                                                                                 | ≤ 25 phút với parquet + dtype xuống float32/int8 + aggregate trên từng chunk                                | dtype ép xuống float32/int8 và đọc theo chunk là kỹ thuật chuẩn                                                                                       | P5, P3         | 30/11/2026 (hết W8)                                            |
| **G6** | Biến kết quả thành**quyết định có chi phí** và **giải trình được**                                            | Ma trận chi phí sai lầm; ngưỡng duyệt đề xuất; bảng 10 driver lớn nhất (hệ số chuẩn hoá + WOE + SHAP); kiểm định công bằng theo 3 nhóm nhân khẩu                       | Không có scorecut-line; không giải trình               | 1 ngưỡng PD\* được đề xuất kèm **đường cong lợi nhuận–tỷ lệ duyệt**; top-10 driver; chênh lệch tỷ lệ duyệt giữa 2 nhóm đối chứng < **5 điểm %** | Đề xuất kèm bảng chính sách theo 4 phân khúc PD và tính được giá trị tiền tệ                    | Mục 1.5 đã chứng minh ngưỡng suy từ chi phí gần tối ưu (lệch 0,007 điểm)                                                                         | P4, P7, P8     | 28/12/2026 (báo cáo cuối)                                    |

**Diễn giải theo đúng 5 tiêu chí SMART** (lấy G1 làm mẫu; các mục tiêu khác có cấu trúc tương tự trong bảng trên):

- **S** – Cụ thể: *mô hình phân loại nhị phân dự báo `TARGET`*, trên *một mẫu xác định*, với *một tập đặc trưng xác định*.
- **M** – Đo lường: ROC-AUC & KS, tính trên fold kiểm định của GroupKFold (5 fold, chia theo nhóm `SK_ID_CURR`), báo kèm khoảng tin cậy bootstrap 95% để chống "ăn may".
- **A** – Khả thi: không đòi hỏi phần cứng lớn, không đòi hỏi thuật toán ngoài tầm; chuẩn so sánh lấy từ chính mặt bằng đã công bố của bộ dữ liệu (G1 đặt ở 0,76–0,78 thay vì 0,81+ vì 0,81 cần feature engineering cực lớn).
- **R** – Relevancy: đánh thẳng vào P1 (nợ xấu) và P3 (chất lượng quyết định); AUC là chỉ tiêu mà hội đồng rủi ro và chính cuộc thi gốc dùng.
- **T** – Hữu hạn: 14/12/2026, khớp mốc hết tuần 10; nếu chưa đạt thì chỉ còn 2 tuần điều chỉnh, đủ để kích hoạt kế hoạch B (mục 6.3).

**Chỉ tiêu KHÔNG đặt (non-goals) — khai báo để phản biện không hỏi thừa**

1. Không đặt chỉ tiêu **accuracy** (sẽ đạt >91% một cách tầm thường) và không cam kết "dự đoán đúng X% khách hàng".
2. Không đặt mục tiêu đưa ra **mô hình tốt nhất thế giới** trên Kaggle; mục tiêu là phương pháp luận minh bạch, tái lập được.
3. Không suy luận **nhân quả** từ hệ số mô hình.
4. Không ngoại suy kết luận sang thị trường Việt Nam về mặt *định lượng*; ngoại suy chỉ ở mức *phương pháp*.

---

## 3. CÂU HỎI NGHIÊN CỨU

Năm câu hỏi cốt lõi, mỗi câu đi kèm giả thuyết kiểm định được, phép thử và dữ liệu cần dùng. Trật tự câu hỏi phản ánh trật tự phụ thuộc: RQ2 chỉ trả lời được khi G5 (kết nối đúng) hoàn tất.

### RQ1 · Trần năng lực dự báo của bài toán nằm ở đâu, và thành phần nào của thông tin tạo nên năng lực đó?

*Câu hỏi nền: trước khi hỏi "mô hình nào tốt hơn", phải biết bài toán này dự báo được đến đâu.*

- **Giả thuyết H1:** thông tin đã được chuẩn hoá trong hồ sơ (đặc biệt `EXT_SOURCE_1/2/3` — điểm từ công ty tín dụng) mang phần lớn năng lực dự báo; AUC của riêng bảng hồ sơ vào khoảng 0,72–0,74.
- **Phép thử:** LogReg hồi quy logistic đa biến + RF trên tập đặc trưng tối giản (10 biến) và đầy đủ (121 biến); so AUC; vẽ biến đơn lẻ tốt nhất (đường cong ROC theo `EXT_SOURCE_3`).
- **Dữ liệu:** `application_train.csv` đơn bảng.
- **Sản phẩm:** biểu đồ "một mình biến nào mạnh nhất"; bảng IV/WOE 121 biến; trần AUC tham chiếu của cấu hình.

### RQ2 · Giá trị gia tăng của dữ liệu hành vi đa bảng là bao nhiêu, và bảng nào đáng giá nhất?

*Câu hỏi trung tâm về mặt kỹ thuật của đề tài — và là chỗ duy nhất chứng minh `merge`/`groupby` có ý nghĩa phương pháp luận chứ không phải bài tập lập trình.*

- **Giả thuyết H2:** các biến tổng hợp từ *lịch sử trả nợ* (`installments_payments`, `POS_CASH_balance`, `credit_card_balance`) cải thiện AUC nhiều hơn *hồ sơ nhân thân*, do giảm bất đối xứng thông tin sau khi đã từng cho vay. Dự kiến ΔAUC ≥ +0,02 so với RQ1.
- **Giả thuyết H2b:** tỷ lệ kỳ hạn trả **đúng hạn** (không phải số ngày quá hạn tuyệt đối) là biến mạnh nhất trong mọi đặc trưng được tạo.
- **Phép thử:** thí nghiệm loại bỏ có kiểm soát — A₀ (chỉ hồ sơ) · A₁=+bureau · A₂=+previous_application · A₃=+POS · A₄=+installments · A₅=+credit_card, và tổ hợp đầy đủ; 5-fold GroupKFold; bootstrap CI 95% cho từng ΔAUC; DeLong-test/so sánh phi tham số nếu cần.
- **Dữ liệu:** toàn bộ 6 bảng.
- **Sản phẩm:** bảng phân bổ đóng góp theo nguồn dữ liệu → **chuyển thẳng thành khuyến nghị đầu tư hệ thống** (nên ưu tiên tích hợp nguồn nào).

### RQ3 · Logistic Regression hay Random Forest — sự đánh đổi giữa sức mạnh dự báo, chi phí vận hành và khả năng giải trình là bao nhiêu?

- **Giả thuyết H3:** RF vượt LogReg về AUC nhưng **không** vượt quá 0,03–0,04; ngược lại LogReg thắng về tốc độ suy luận, dung lượng và khả năng giải trình bằng hệ số/WOE.
- **Giả thuyết H3b:** chênh lệch hiệu năng chủ yếu đến từ **tương tác và phi tuyến** (đặc biệt giữa DTI, thu nhập và số người phụ thuộc), có thể thu hẹp bằng LogReg + biến tương tác đã được thiết kế thủ công.
- **Phép thử:** cùng tập đặc trưng, cùng split, cùng seed; báo AUC/AP/KS/log-loss/thời gian fit/suy luận/dung lượng bộ nhớ; độ lệch hiệu năng giữa 2 mô hình trên **từng phân khúc** (thu nhập, khu vực, hợp đồng). LogReg dùng `C` điều chuẩn + penalty L1 để chọn biến; RF dùng `max_features`/`min_samples_leaf`/`class_weight` qua randomized search.
- **Sản phẩm:** bảng đánh đổi 4 cột + biểu đồ; khuyến nghị mô hình cho môi trường có ràng buộc pháp lý về giải trình.

### RQ4 · Kỹ thuật xử lý mất cân bằng nào cải thiện chỉ tiêu **kinh doanh** mà không phá vỡ ý nghĩa xác suất?

- **Giả thuyết H4:** resample (SMOTE/undersample) cải thiện AP/Recall nhưng **làm sai lệch calibration** → PD̂ bị thổi phồng, kéo theo scorecut-line sai và cắt nhầm khách tốt.
- **Giả thuyết H4b:** chỉ cần **dịch ngưỡng quyết định theo chi phí** (mục 1.5) là thu được phần lớn lợi ích của resample mà vẫn giữ nguyên PD — giải pháp được ưu tiên.
- **Phép thử:** 5 phương án (a)–(e) ở G4, áp dụng **chỉ trên tập huấn luyện của mỗi fold** (chống rò rỉ); đo đồng thời AP, Recall@10%, **Brier**, độ dốc hiệu chuẩn, và **lợi nhuận kỳ vọng tại ngưỡng p\***.
- **Sản phẩm:** "bảng 5×6" chọn phương án; khuyến nghị thực hành cho hội đồng rủi ro (mô hình huấn luyện bằng class_weight nhưng **hiệu chuẩn lại bằng Platt/isotonic** nếu cần).

### RQ5 · Ngưỡng duyệt nào tối đa hoá lợi nhuận kỳ vọng, và nó tạo ra hệ luỵ phân bổ (công bằng/tài chính toàn diện) thế nào?

*Câu hỏi kết nối thống kê → chính sách; quyết định tính "tài chính – ngân hàng" của đề tài.*

- **Giả thuyết H5:** ngưỡng cắt suy từ chi phí (`p* = 20%`) gần tối ưu; nhưng **nếu** dùng mốc 0,5 (mặc định sklearn) thì danh mục sẽ bị thắt quá mức, mất doanh thu không cần thiết.
- **Giả thuyết H5b (công bằng):** nếu chỉ dùng biến nhân khẩu học, mô hình có thể tạo chênh lệch tỷ lệ duyệt theo khu vực/giới; nếu thêm biến hành vi, chênh lệch này **giảm** vì rủi ro được giải thích bằng hành vi chứ không bằng xuất thân.
- **Phép thử:** đường cong lợi nhuận & nợ xấu theo approval rate (Hình 2); phân tích decile; demographic gap và equalized-odds giữa các nhóm `CODE_GENDER`, `NAME_FAMILY_STATUS`, `REGION_RATING_CLIENT`, nhóm tuổi; kiểm định xem sự khác biệt còn lại có được giải thích bởi tỷ lệ vỡ nợ thực tế trong nhóm hay không.
- **Sản phẩm:** **bảng chính sách duyệt vay** (ngưỡng → tỷ lệ duyệt → nợ xấu kỳ vọng → lợi nhuận kỳ vọng) + mục "hàm ý chính sách & giới hạn đạo đức" trong báo cáo.

### RQ6 · (Mở rộng, nếu còn thời gian) Chệch chọn mẫu do "sống sót" ảnh hưởng bao nhiêu, và `previous_application` có sửa được không?

- **Giả thuyết H6:** mẫu chỉ chứa hồ sơ *đã được duyệt*, nên PD của toàn bộ quần thể xin vay bị **đánh giá thấp**; dùng `NAME_CONTRACT_STATUS = Refused/Unused/Distribution` làm nhóm đối chứng qua **reject inference** (mở rộng mẫu 1,67 triệu đơn) sẽ làm tăng PD ước lượng ở vùng biên.
- **Phép thử:** gán nhãn giả định cho đơn bị từ chối (parceling / augmentation), so sánh độ dốc của hàm PD ở vùng biên; báo cả kết luận nếu không đủ bằng chứng.
- **Sản phẩm:** mục "giới hạn của nghiên cứu" — phần thường bị bỏ sót nhất trong đồ án, và là phần hội đồng đánh giá cao nhất.

> **Ghép câu hỏi ↔ dữ liệu ↔ phương pháp:** ma trận truy xuất ở mục 5 bảo đảm không có câu hỏi nào "mồ côi" (không có phép kiểm định) và không có biến nào "tự phát" (không phục vụ câu hỏi nào).

---

## 4. BIẾN MỤC TIÊU (Y) VÀ CÁC BIẾN GIẢI THÍCH (X)

### 4.1. Nhận dạng bản chất bài toán: tại sao là Phân loại, không phải Hồi quy hay Phân cụm

| Cách tiếp cận                                | Nhãn cần có                                         | Dữ liệu này có phù hợp?                                                                                                                                                                                     | Phán quyết                                                                                                                                                                                            |
| ----------------------------------------------- | ------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Phân loại nhị phân có giám sát** | Có sẵn 2 lớp`TARGET ∈ {0,1}` cho 307 511 hồ sơ | ✅ Hoàn toàn khớp: nhãn tồn tại, rời rạc, 2 giá trị, đúng đơn vị quyết định (1 hồ sơ)                                                                                                         | ✅**CHỌN** — bài toán *Probability of Default*                                                                                                                                              |
| **Hồi quy (Regression)**                 | Nhãn liên tục                                       | ❌ Không có nhãn liên tục. Có thể dùng làm*cách trình bày tương đương*: hồi quy trên Y ∈ {0,1} = ước lượng PD (Linear Probability Model) — nhưng vi phạm [0,1] và cho dự báo kém | ⚠️ Loại.*Ngoại lệ:* nếu mở rộng sang **LGD** (số tiền mất, cần dữ liệu thu hồi nợ — không có) hoặc **số ngày quá hạn tương lai** thì đó mới là hồi quy    |
| **Phân cụm không giám sát**          | Không cần nhãn                                      | ⚠️ Làm được (KMeans/HCA để phân khúc khách hàng) nhưng**không trả lời câu hỏi kinh doanh** "ai sẽ vỡ nợ", và không đo được bằng AUC                                             | ⚠️**Loại làm bài toán chính**; được dùng làm *công cụ phụ*: phân khúc khách hàng để đặt ngưỡng theo từng cụm, và kiểm tra tính ổn định của nhóm đặc trưng |
| **Survival analysis (time-to-default)**   | Nhãn = thời điểm + trạng thái                    | ⚠️ Bộ này chỉ cho biết*có/không* vỡ nợ trong cửa sổ cố định, không cho ngày vỡ nợ chính xác                                                                                                | ❌ Ngoài phạm vi (ghi trong phần giới hạn)                                                                                                                                                         |

**Ba lập luận kỹ thuật bảo vệ lựa chọn phân loại nhị phân:**

1. **Đơn vị ra quyết định là nhị phân** (duyệt/từ chối) ⇒ mô hình phải tối ưu việc **xếp hạng rủi ro** giữa hai trạng thái, tức tối ưu ROC/PR, không tối ưu sai số bình phương.
2. **Nhãn mất cân bằng 1 : 11,4** ⇒ phải dùng hàm mất mát/log-loss có trọng số, metric PR-AUC và KS; nếu là hồi quy, các kỹ thuật này mất chỗ dựa.
3. **Deliverable thật là xác suất, không phải nhãn.** Vì ngưỡng quyết định *không phải* 0,5 mà là `p* = 20%` (mục 1.5), mô hình phân loại bắt buộc phải **hiệu chuẩn xác suất**. Đây là điểm khác biệt giữa "làm phân loại cho đúng giáo trình" và "làm phân loại để dùng được trong ngân hàng".

> **Câu trả lời một câu cho hội đồng:** *Đây là bài toán **phân loại nhị phân có giám sát**, trong đó đầu ra được dùng như một **xác suất vỡ nợ được hiệu chuẩn** (scorecard IRB-style), chứ không dùng nhãn rời rạc; và **không** phải hồi quy vì nhãn không liên tục, **không** phải phân cụm vì có sẵn nhãn và cần dự báo chứ không cần khám phá cấu trúc.*

### 4.2. Biến mục tiêu Y

| Thuộc tính                              | Mô tả                                                                                                                                                                   |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Tên / nguồn**                   | `TARGET`, duy nhất trong `application_train.csv` (không có trong `application_test.csv`)                                                                         |
| **Kiểu**                           | Nhị phân:**1 = khách hàng không trả được nợ** (vỡ nợ), **0 = trả đủ**                                                                          |
| **Tần suất [K]**                  | 24 825 hồ sơ nhãn 1 / 307 511 =**8,0729%**; 282 686 hồ sơ nhãn 0 → tỷ lệ **11,39 : 1**                                                               |
| **Cửa sổ quan sát**              | Theo tài liệu cuộc thi: khả năng trả các kỳ hạn trong**12 tháng** kể từ kỳ đầu sau giải ngân (khoảng 06/2016)                                     |
| **Grain**                           | 1 hồ sơ vay (một hợp đồng), không phải 1 khách hàng — một khách*có thể* có nhiều hợp đồng → phải kiểm tra trùng `SK_ID_CURR` ở Giai đoạn 2 |
| **Tính đối xứng của sai lầm** | FN (bỏ sót khách xấu) đắt hơn FP (từ chối oan khách tốt) khoảng**4 : 1** với cấu trúc chi phí ở mục 1.5 (`LGD/r = 0,60/0,15`)                   |

**Định nghĩa vận hành "vỡ nợ" (điều một đề tài tài chính phải nêu rõ, điều một đề tài data science thường bỏ qua).**
Bộ dữ liệu không công bố ngưỡng DPD; do vậy tài liệu này định nghĩa theo thông lệ quản trị Việt Nam để đối chiếu được với số liệu nội bảng:

- **Default (Y = 1):** khoản vay chuyển vào **nhóm 5** — quá hạn **> 90 ngày** — tại bất kỳ thời điểm nào trong cửa sổ 12 tháng, **hoặc** bị phân loại nợ xấu nhóm 3–5 theo quy định phân loại nợ hiện hành và không hồi phục trong 180 ngày.
- **Quy tắc "back-in-cure":** một hồ sơ chỉ được coi là vỡ nợ *ổn định* nếu vi phạm ngưỡng trong **≥ 2 kỳ liền kề trong 6 kỳ** (hoặc 3 trong 12 kỳ). Nếu dùng ngay lần trễ hạn đầu tiên, Y sẽ bao gồm cả "trễ hạn tạm thời" — nhãn nhiễu, AUC giảm, và quan trọng hơn, **mâu thuẫn với cách ngân hàng trích lập dự phòng**.
- **Không nhầm 3 khái niệm:** *chậm trả (delinquency, DPD 1–89)* ≠ *vỡ nợ (default, >90 ngày)* ≠ *mất vốn (loss/write-off)*. Đề tài dự báo **default**; **loss** là LGD (ngoài phạm vi).

**Ba hệ quả phương pháp luận của việc chọn Y:**

1. **Y chỉ "chín" sau 12 tháng** ⇒ mọi biến đưa vào phải được quan sát **tại hoặc trước ngày nộp đơn**. Đây là ranh giới chống **rò rỉ thời gian (temporal leakage)** — sai lầm phổ biến nhất khi agg từ bảng phụ (ví dụ dùng `DAYS_CREDIT_UPDATE` hoặc trạng thái khoản vay *sau* ngày duyệt).
2. **Không có biến nào trong X được phép "biết trước" tương lai**; nếu một biến có AUC đơn biến > 0,9, nghi ngờ đầu tiên phải là leakage chứ không phải "phát hiện lớn".
3. Nhãn chỉ tồn tại trên quần thể **đã được duyệt** ⇒ bias chọn mẫu (RQ6).

### 4.3. Khoá phương pháp luận: "grain discipline"

`Y` và `X` chỉ so sánh được nếu cùng grain. Toàn bộ đề tài tuân thủ bất biến:

```
∀ bảng phụ:   df_agg = df.groupby("SK_ID_CURR").agg([...])   # n dòng → 1 dòng
              assert df_agg.index.is_unique                    # 1 hồ sơ : 1 dòng
              train = train.merge(df_agg, on="SK_ID_CURR", how="left")
              assert len(train) == 307_511                     # mẫu KHÔNG đổi kích thước
              assert train.TARGET.notna().all()
```

Nếu `len(train)` sau merge ≠ 307 511 thì phép nối đã làm **nở dòng (fan-out)** và mọi chỉ số phía sau đều vô nghĩa. Đây là điểm kiểm tra bắt buộc ở Giai đoạn 3 (xem Hình 1).

![Sơ đồ liên kết dữ liệu](assets/fig1_data_model.png)

### 4.4. Khoá biến giải thích X: 12 khối đặc trưng

Quy ước ký hiệu: **[1]** = chỉ từ `application_train`; **[2+]** = phải `groupby().agg()` trên bảng phụ rồi merge. Cột "Kỳ vọng" là dấu của liên hệ dự kiến với vỡ nợ để kiểm chứng ở RQ1–RQ2 (**không** phải kết luận).

| Khối                                            | Bảng nguồn                                                                               | Biến gốc tiêu biểu                                                                                                                                                                                                                                                                 | Kỹ thuật                                                                                                                                                                        | Kỳ vọng dấu với vỡ nợ                                                                                                         |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| **X1. Nhân thân & hộ tịch**            | [1]                                                                                        | `CODE_GENDER`, `NAME_FAMILY_STATUS`, `CNT_CHILDREN`, `CNT_FAM_MEMBERS`, `NAME_EDUCATION_TYPE`                                                                                                                                                                                | One-hot / ordinal;`CNT_CHILDREN/CNT_FAM_MEMBERS` → tỷ lệ phụ thuộc                                                                                                         | ± yếu, dễ gây bias ⇒**kiểm định công bằng**                                                                         |
| **X2. Thu nhập & gánh nặng nợ**        | [1]                                                                                        | `AMT_INCOME_TOTAL`, `AMT_CREDIT`, `AMT_ANNUITY`, `AMT_GOODS_PRICE`                                                                                                                                                                                                             | Tỷ lệ + winsorize + log                                                                                                                                                         | **DTI >** mạnh (kỳ vọng dương); `AMT_INCOME_TOTAL` âm                                                                 |
| **X3. Loại khoản vay & mục đích**     | [1]                                                                                        | `NAME_CONTRACT_TYPE` (Cash/Revolving), `NAME_CASH_LOAN_PURPOSE`, `NAME_GOODS_CATEGORY` (từ prev)                                                                                                                                                                                | Ordinal/target-encoding có CV an toàn                                                                                                                                           | "vay tiêu dùng không mục đích rõ" dương                                                                                    |
| **X4. Việc làm & tổ chức**             | [1]                                                                                        | `OCCUPATION_TYPE`, `ORGANIZATION_TYPE`, `NAME_INCOME_TYPE`, `DAYS_EMPLOYED`, `EMP_TYPE` (`..._WORK_*`)                                                                                                                                                                     | Gộp hiếm (≥0,5% mẫu);`DAYS_EMPLOYED` xử lý mã lỗi (4.6)                                                                                                                 | Thâm niên âm; "Commercial" dương;`Unemployed`/`Maternity` dương                                                          |
| **X5. Cư trú & địa bàn**              | [1]                                                                                        | `NAME_HOUSING_TYPE`, `REGION_POPULATION_RELATIVE`, `REGION_RATING_CLIENT(_W_CITY)`, `DAYS_REGISTRATION`, các `*_NOT_*` (lệch khu vực)                                                                                                                                     | `flag` rời rạc + phân vị region                                                                                                                                             | Rating 2–3 dương; lệch nơi ở/nơi làm dương                                                                                |
| **X6. Điểm từ bên thứ ba**            | [1]                                                                                        | `EXT_SOURCE_1/2/3`, `FLAG_MOBIL/EMP_PHONE/PHONE/EMAIL`, `DAYS_LAST_PHONE_CHANGE`                                                                                                                                                                                                 | Giữ nguyên + missing-indicator                                                                                                                                                  | **Mạnh nhất đơn biến** (EXT_SOURCE âm)                                                                                  |
| **X7. Cường độ bị thẩm định**      | [1]                                                                                        | `AMT_REQ_CREDIT_BUREAU_HOUR/DAY/WEEK/MONTH/YEAR` (số lượt CIC hỏi trong 1 năm)                                                                                                                                                                                                  | Tổng + log; "hỏi nhiều, vay nhiều"                                                                                                                                            | Dương mạnh (cầu vốn cấp bách)                                                                                                |
| **X8. Bộ chứng từ**                     | [1]                                                                                        | `FLAG_DOCUMENT_2…18` (3-NDFL, sao kê, hộ chiếu, quân đội…)                                                                                                                                                                                                                   | Đếm số chứng từ; gộp 17 cờ → 1`N_DOC`                                                                                                                                   | Dương yếu (hồ sơ phải chứng minh = rủi ro cao hơn)                                                                         |
| **X9. Môi trường sống (vĩ mô)**      | [1]                                                                                        | `BASEMENTAREA_AVG`, `FONDKAPREMONT_MODE`, `MODELS_AVG/MODE_SUM`, `ELEVATORS_MEDI`, `NONSTANDARD…`                                                                                                                                                                           | **Chỉ dùng tổng hợp** (PCA hoặc 1 index) — missing 50–70% [C]                                                                                                        | Âm (khu tốt hơn = rủi ro thấp hơn)                                                                                            |
| **X10. Lịch sử tại TCTD khác**         | **bureau** + bureau_balance                                                          | `CREDIT_ACTIVE`, `CREDIT_TYPE`, `DAYS_CREDIT`, `CREDIT_DAY_OVERDUE`, `AMT_CREDIT_SUM{,_DEBT,_LIMIT,_OVERDUE}`, `CNT_CREDIT_PROLONG`, `AMT_CREDIT_MAX_OVERDUE` → `STATUS` (0–5) từ bureau_balance                                                                    | **1:n → agg 3 tầng**: tháng → khoản (`SK_ID_BUREAU`) → hồ sơ; count/mean/max/sum/std + share trạng thái                                                         | Số khoản dương;**tỷ lệ quá hạn dương mạnh**; `CREDIT_ACTIVE=Bad debt` dương                                    |
| **X11. Lịch sử quan hệ với chính NH** | **previous_application**                                                             | `NAME_CONTRACT_STATUS` (Approved/Refused/Unused/Canceled), `AMT_APPLICATION` vs `AMT_CREDIT` (chênh đề xuất–duyệt), `AMT_DOWN_PAYMENT`, `RATE_DOWN_PAYMENT`, `DAYS_DECISION`, `CODE_REJECT_REASON`, `CHANNEL_TYPE`, `NAME_CLIENT_TYPE`, `CREDIT_DAY_OVERDUE` | Đếm đơn / tỷ lệ bị từ chối / số ngày kể từ đơn cuối /**avg (AMT_CREDIT − AMT_APPLICATION)** = độ "vượt hạn mức được duyệt"                       | Tỷ lệ đơn bị từ chối dương; khách cũ (`Repeat`) âm; thời gian từ đơn cuối âm                                    |
| **X12. Hành vi trả nợ chi tiết**       | **POS_CASH_balance**, **installments_payments**, **credit_card_balance** | `SK_DPD`, `SK_DPD_DEF`, `MONTHS_BALANCE`, `NAME_CONTRACT_STATUS`; `AMT_INSTALMENT` vs `AMT_PAYMENT`, `NUM_INSTALMENT_NUMBER` vs `CNT_INSTALMENT(_FUTURE)`; `NAME_CURRENCY`, `AMT_BALANCE` vs `AMT_CREDIT_LIMIT_AMOUNT`, `AMT_DRAWINGS_ATM_CURRENT`             | Kỳ hạn→khoản→hồ sơ.**Các tỷ lệ** là trung tâm: % kỳ hạn trả đúng, % trả thiếu, số dư còn phải trả, **utilization** thẻ, số tháng có DPD>0 | **Mạnh nhất toàn bộ mô hình**: DPD dương; `CNT_INSTALMENT_FUTURE` (nợ còn lại) dương; utilization thẻ dương |

**Ghi chú phân bổ "công sức":** X1–X9 nằm sẵn trong 1 bảng nên chỉ cần làm sạch + mã hoá; **X10–X12 mới là chỗ đề tài tạo giá trị** và cũng là chỗ duy nhất kiểm chứng kỹ thuật `groupby().agg()` mà đề bài yêu cầu. Mọi rủi ro kỹ thuật (RAM, fan-out, leakage) cũng nằm hết ở X10–X12.

### 4.5. Mười lăm biến phái sinh trọng tâm (spec chi tiết, dùng trực tiếp ở Giai đoạn 3)

| #   | Tên biến                  | Công thức (pandas)                                                                                             | Bảng        | Logic kinh tế                                                   |
| --- | --------------------------- | ---------------------------------------------------------------------------------------------------------------- | ------------ | ---------------------------------------------------------------- |
| F1  | `DTI`                     | `AMT_ANNUITY / (AMT_INCOME_TOTAL/12)`                                                                          | [1]          | Trả nợ/thu nhập tháng — chỉ tiêu số 1 trong thẩm định |
| F2  | `CREDIT_BURDEN`           | `AMT_ANNUITY / AMT_CREDIT`                                                                                     | [1]          | Cường độ trả gốc+lãi trên quy mô vay                    |
| F3  | `LTV`                     | `AMT_CREDIT / AMT_GOODS_PRICE`                                                                                 | [1]          | Cho vay vượt giá trị tài sản ⇒ rủi ro & đạo đức      |
| F4  | `INCOME_PER_MEMBER`       | `AMT_INCOME_TOTAL / CNT_FAM_MEMBERS`                                                                           | [1]          | Thu nhập thực của hộ                                         |
| F5  | `OVERDUE_BURDEN`          | `AMT_CREDIT_SUM_DEBT / AMT_INCOME_TOTAL`                                                                       | bureau       | Nợ toàn thị trường / thu nhập                              |
| F6  | `UTIL_CURRENT`            | `ΣAMT_BALANCE / ΣAMT_CREDIT_LIMIT_AMOUNT`                                                                    | credit_card  | Tỷ lệ dùng hạn mức (điểm FICO-style)                      |
| F7  | `MAX_DPD_ACTIVE`          | `max(SK_DPD)` chỉ trên các tháng còn hiệu lực (`SK_DPD_DEF` đã loại trễ hạn đã trả trong kỳ) | POS + CC     | Trễ hạn nặng nhất từng xảy ra                              |
| F8  | `DPD_MONTH_RATIO`         | `count(SK_DPD>0) / count(số tháng quan sát)`                                                                | POS + CC     | Tần suất trễ hạn, không chỉ cực đại                     |
| F9  | `PAYMENT_RATIO`           | `mean(AMT_PAYMENT / AMT_INSTALMENT)`                                                                           | installments | Trả đủ hay trả kiểu "đối phó"                            |
| F10 | `SHORTFALL_SUM`           | `Σ max(AMT_INSTALMENT − AMT_PAYMENT, 0)`                                                                     | installments | Số tiền thiếu luỹ kế                                        |
| F11 | `EARLY_PAY_RATIO`         | `mean(DAYS_INSTALMENT − DAYS_ENTRY_PAYMENT > 0)`                                                              | installments | Trả trước hạn = tín hiệu tốt                              |
| F12 | `DUE_DILIGENCE_INTENSITY` | `Σ AMT_REQ_CREDIT_BUREAU_{HOUR..YEAR}`                                                                        | [1]          | Bao nhiêu tổ chức đang soi hồ sơ này                      |
| F13 | `REJECT_RATIO`            | `count(NAME_CONTRACT_STATUS=='Refused') / count(đơn)`                                                        | previous_app | Lịch sử bị từ chối = thị trường đã cảnh báo          |
| F14 | `RECENCY`                 | `min(abs(DAYS_DECISION))`                                                                                      | previous_app | Quan hệ gần đây nhất                                        |
| F15 | `BAD_DEBT_SHARE`          | `count(CREDIT_ACTIVE=='Bad debt') / count(khoản)`                                                             | bureau       | Vết đen trong CIC                                              |

> Nguyên tắc thiết kế: **ưu tiên tỷ lệ hơn số tuyệt đối** (khử quy mô hồ sơ), **luôn kèm bộ đếm** (100% đúng hạn trên 2 kỳ ≠ 100% trên 40 kỳ), và **mọi_agg đều phải có missing-indicator** vì khách hàng "không có lịch sử" là một trạng thái kinh tế có ý nghĩa (thin file), không phải chỗ trống cần điền.

### 4.6. Sổ đăng ký chất lượng dữ liệu (Data Quality Register) — các bẫy đã biết của bộ dữ liệu

Nhãn **[K/C]**: đặc điểm được cộng đồng phân tích bộ dữ liệu ghi nhận rộng rãi, **phải xác nhận lại bằng số đo cụ thể ở Giai đoạn 2** trước khi viết vào báo cáo.

| #   | Hiện tượng                                                                                                                                         | Nhận diện                                             | Xử lý                                                                                                                                  | Nếu bỏ qua                                                                                    |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Q1  | **`DAYS_EMPLOYED = 365243` (số dương)** — mã nội bộ cho "không xác định/không làm việc", ≈5,8% [C]                             | `df.DAYS_EMPLOYED > 0`                                | Tạo cờ`FLAG_UNK_EMPLOYMENT`, đặt giá trị về NaN **sau** khi tách cờ; **không** log, **không** trung bình | Biến "thâm niên" bị lệch 1000 năm, mọi tỷ lệ phái sinh sai                            |
| Q2  | Toàn bộ biến`DAYS_*` **âm = quá khứ**                                                                                                   | `min()` âm                                           | `abs()` *sau khi* tách trường hợp Q1; luôn ghi rõ chiều thời gian trong bảng biến                                          | Dấu sai → "khách trẻ hơn" hoá "già hơn"                                                 |
| Q3  | `AMT_INCOME_TOTAL` & `AMT_ANNUITY` có outlier cực đại (hàng trăm triệu) [C]                                                                | `describe()`, quantile 99,5                           | Winsorize 1%/99% hoặc log1p;**giữ NaN thay vì median toàn cục**                                                               | RF ít ảnh hưởng nhưng**LogReg sai nghiêm trọng** (chuẩn hoá dựa trên mean/var) |
| Q4  | `AMT_CREDIT_SUM_DEBT = 0` ở khoản `Closed` — **0 thật hay thiếu?**                                                                     | cross-tab với`CREDIT_ACTIVE`                         | Chỉ tính nợ còn lại trên Active; tạo`n_closed_credits`                                                                          | UNDERestimate gánh nặng nợ của khách có nhiều khoản tất toán                          |
| Q5  | `CREDIT_CURRENCY = "currency B"` — đơn vị khác xen lẫn trong bureau [C]                                                                       | `value_counts()`                                      | Lọc hoặc quy đổi;**tuyệt đối không sum lẫn đơn vị**                                                                    | `AMT_CREDIT_SUM` sai thứ tự độ lớn                                                       |
| Q6  | `bureau.csv` có **`SK_ID_BUREAU` trùng lặp** [C]                                                                                         | `duplicated().sum()`                                  | `drop_duplicates()` trước khi nối bureau_balance                                                                                    | 27M dòng bị nhân đôi, RAM vỡ, biến DPD sai                                               |
| Q7  | `bureau_balance.STATUS` có giá trị **`X`** (không rõ) và **`C`** (đóng)                                                     | `value_counts()`                                      | Tách thành 2 trạng thái riêng, đừng ép về số                                                                                   | Gộp`X` vào "không quá hạn" → bỏ sót rủi ro                                           |
| Q8  | `installments_payments`: **nhiều dòng cho 1 kỳ hạn** (trả nhiều lần)                                                                   | `groupby([SK_ID_PREV, NUM_INSTALMENT_NUMBER]).size()` | **gộp lên cấp kỳ hạn trước**, rồi mới gộp lên cấp khoản                                                               | Double-count tiền trả,`PAYMENT_RATIO` > 1 hàng loạt                                       |
| Q9  | Missing 50–70% ở khối X9 (môi trường sống) và 20–50% ở`EXT_SOURCE_1` [C]                                                                  | `isna().mean().sort_values()`                         | Drop cột >60% missing; giữ cột mạnh +**missing indicator**                                                                     | RF cây yếu vì phải cắt ở cột toàn NaN                                                   |
| Q10 | Nhóm`AMT_*` có số **0** thay vì NaN ("0 nghĩa là không khai" hay "bằng 0"?)                                                           | `==0` per biến                                       | Đưa 0 → NaN với các biến mà 0 phi thực tế (thu nhập, giá trị hàng)                                                          | 0 kéo trung bình, tạo "khách hàng 0 đồng"                                                |
| Q11 | **Rò rỉ thời gian** qua `DAYS_CREDIT_UPDATE`/trạng thái sau ngày duyệt; và qua_agg dùng dữ liệu *sau* `DAYS_DECISION`          | kiểm tra`DAYS_* < 0` trong cửa sổ                  | Mọi agg giới hạn`DAYS_x < 0` so với ngày nộp đơn                                                                               | AUC 0,85 "ảo" rồi sập khi back-test — kịch bản tồi tệ nhất                             |
| Q12 | `OCCUPATION_TYPE`/`ORGANIZATION_TYPE` **~25–30% missing + 50–100 giá trị**                                                              | `nunique()`                                           | Gộp nhóm hiếm (<0,5%) vào`Other`; hoặc target-encoding có CV                                                                     | One-no làm ma trận nở tới 400+ cột mỏng, ít thông tin, RF chọn biến nhiễu            |
| Q13 | **`SK_ID_CURR` không chắc duy nhất** giữa train/test cùng một khách hàng (1 khách nhiều hợp đồng)                                | đếm trùng trên`SK_ID_*` khác                     | `GroupKFold(groups=SK_ID_CURR)`; kiểm tra overlap với test                                                                           | Overfit theo khách hàng, đánh giá lạc quan                                                |
| Q14 | Nhãn 1 chỉ 8,07% → sai số chuẩn của AUC ~0,003 trên test; sự khác biệt 0,005 AUC giữa hai mô hình**có thể không có ý nghĩa** | bootstrap CI                                            | Luôn báo CI 95%, kiểm định so sánh cặp                                                                                            | Kết luận "RF tốt hơn LogReg" dựa trên nhiễu                                              |

### 4.7. Bộ chỉ tiêu đánh giá — và vì sao chọn chúng

| Chỉ tiêu                                              | Vì sao dùng trong bài toán này                                                                                                                              | Vai trò ra quyết định | Ngưỡng                           |
| ------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------- | ---------------------------------- |
| **ROC-AUC**                                       | Miễn nhiễm với điểm cắt và (tương đối) với tỷ lệ nền; là metric của cuộc thi gốc & của hội đồng rủi ro khi xét*khả năng xếp hạng* | Chỉ tiêu chính của G1 | ≥ 0,76 / 0,78                     |
| **KS statistic**                                  | Ngôn ngữ chuẩn của scorecard ngân hàng (max\|TPR − FPR\|) → "danh mục có tách được không"                                                         | Chỉ tiêu chính của G1 | ≥ 0,40                            |
| **Gini = 2·AUC−1**                              | Cách ngành bảo hiểm/ngân hàng quen đọc                                                                                                                   | Trình bày               | ≥ 0,52                            |
| **PR-AUC (AP)**                                   | Có ý nghĩa dưới 8% dương tính; ROC-AUC "đẹp" mà AP ≈ π là mô hình vô dụng về kinh tế                                                         | Chỉ tiêu chính của G2 | ≥ 0,28                            |
| **Recall @ top-10%/decile-1 lift**                | Khớp đúng câu hỏi vận hành: "bộ lọc đầu tiên bắt được bao nhiêu kẻ xấu"                                                                       | G2                        | Recall ≥ 0,32                     |
| **Brier + calibration slope/intercept**           | **Bắt buộc** vì ngưỡng là `p* = 20%` chứ không phải 0,5: PD phải có nghĩa tuyệt đối                                                       | G4                        | Brier ≤ 0,07; slope ∈ [0,9; 1,1] |
| **Lợi nhuận kỳ vọng tại ngưỡng** (Hình 2) | Bridge duy nhất giữa thống kê và P&L; là thứ CFO đọc được                                                                                            | G6                        | > baseline duyệt-tất-cả         |
| **Accuracy / F1**                                 | Accuracy bị**loại có chủ đích** (dummy đạt 91,93%); F1 báo phụ lục, không quyết định                                                        | —                        | không dùng làm chỉ tiêu       |

**Ma trận chi phí sai lầm để chốt ngưỡng** (điền số ở Giai đoạn 5):

|                                 | Dự đoán: KHÔNG vỡ nợ                                   | Dự đoán: VỠ NỢ (từ chối)                                  |
| ------------------------------- | ------------------------------------------------------------ | ---------------------------------------------------------------- |
| **Thực: không vỡ nợ** | Lợi nhuận`+r` (duyệt đúng)                            | Tổn thất cơ hội`−r` (**FP**: mất một khách tốt) |
| **Thực: vỡ nợ**        | Tổn thất`−LGD` (**FN**: duyệt nhầm khách xấu) | Tránh được`0` (đúng, nhưng cũng mất cơ hội `r`)   |

Ngưỡng cắt tối ưu: `p* = c_FP/(c_FP+c_FN) = r/(r+LGD) = 20%` **[M]**, kiểm chứng chéo bằng đường cong lợi nhuận. Đây là lý do mô hình phân loại trong tài chính **không bao giờ dùng mặc định 0,5**.

### 4.8. Thiết kế kiểm định tính hợp lệ

| Quyết định             | Lựa chọn & lý do                                                                                                                                                                                                                   |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Chia mẫu                 | `train_test_split` stratified theo `TARGET` **nhưng** chỉ để minh hoạ; chỉ tiêu chính đo bằng **GroupKFold(5) trên `SK_ID_CURR`** để tránh cùng một khách hàng rơi vào cả train và test (Q13) |
| Thứ tự biến/mẫu       | Trước khi merge:`sort_values('DAYS_x')` và cắt theo `DAYS < 0` so với ngày nộp đơn để chặn leakage ngược thời gian                                                                                                 |
| Mất cân bằng           | Chỉ**resample tập train trong từng fold** (SMOTE đặt *sau* `train_test_split`, không đặt trước CV) — nếu đặt trước, mẫu tổng hợp rò rỉ sang test và **AUC tăng giả tạo**                     |
| Encoding biến mục tiêu | Target/WOE encoding phải tính**trong fold**, có smoothing, không dùng chính dòng đang encode                                                                                                                            |
| Chọn biến               | L1 (Lasso) cho LogReg + importance OOB permutation cho RF + IV/WOE 121 biến; cắt ở IV < 0,02; báo cáo**biến đã loại** (minh bạch)                                                                                     |
| Siêu tham số            | `RandomizedSearchCV` + `n_iter` cố định, seed 42; **cấm** chọn siêu tham số trên test                                                                                                                               |
| Kiểm định mô hình    | Bootstrap 1000 lần trên test → CI 95% cho AUC/KS (đối phó Q14); so sánh mô hình bằng paired bootstrap                                                                                                                       |
| Ổn định                | PSI giữa phân phối điểm train vs test, và trên`application_test` (48 744 hồ sơ, không nhãn) như một **out-of-sample drift check**                                                                                |

---

## 5. MA TRẬN TRUY XUẤT (traceability matrix)

Không một thành phần nào của đề tài được tồn tại mà không phục vụ một mục tiêu, và không mục tiêu nào thiếu phép đo.

| Mục tiêu          | Câu hỏi NC            | Biến liên quan                        | Kỹ thuật                                                        | Chỉ tiêu đạt                              | Deliverable                                                           | Tuần             |
| ------------------- | ----------------------- | --------------------------------------- | ----------------------------------------------------------------- | --------------------------------------------- | --------------------------------------------------------------------- | ----------------- |
| G1                  | RQ1                     | X1–X9 (đặc biệt X6`EXT_SOURCE_*`) | LogReg + RF, L1, scaling                                          | AUC ≥ 0,78 · KS ≥ 0,42                     | Bảng 4 mô hình × 6 chỉ số + CI                                  | W5–W9            |
| G2                  | RQ1, RQ4                | toàn bộ X (nhìn qua PR)              | PR-AUC, decile/lift table, threshold-free                         | AP ≥ 0,30 · Recall@10% ≥ 0,35              | Biểu đồ lift + decile chart                                        | W9–W10           |
| G3                  | RQ2                     | **X10, X11, X12**                 | `groupby().agg()` → `pd.merge()`, ablation A₀–A₅          | ΔAUC ≥ +0,02; 1 bảng ≥ top-2              | Bảng "giá trị từng nguồn dữ liệu"                              | W3–W4, chốt W10 |
| G4                  | RQ4                     | mọi X (mẫu thay đổi)                | class_weight · NearMiss · SMOTE · threshold                    | AP +0,02**và** Brier ≤ 0,07           | Bảng 5 phương án × 6 chỉ số                                    | W9–W11           |
| G5                  | RQ2 (điều kiện cần) | —                                      | parquet, dtype ép nhỏ, chunked agg, assert grain                | ≤ 45 phút, ≤ 16 GB, 0 lỗi grain           | `build_features.py` + log chạy                                     | W2–W8            |
| G6                  | RQ5, RQ6                | X1, X5 (fairness)                       | cost matrix, đường cong lợi nhuận, SHAP/WOE, demographic gap | 1 ngưỡng + top-10 driver + gap < 5 điểm % | **Bảng chính sách duyệt vay** + mục "Hàm ý & giới hạn" | W11–W12          |
| *(mở rộng)* RQ6 | RQ6                     | `previous_application` (`Refused`)  | reject inference (parceling)                                      | báo cáo định tính + ΔPD vùng biên     | Mục "Giới hạn nghiên cứu"                                        | W12 (optional)    |

---

## 6. KẾ HOẠCH, RỦI RO & TIÊU CHÍ NGHIỆM THU

### 6.1. Lộ trình 12 tuần (06/10/2026 – 28/12/2026)

| Tuần | Ngày        | Giai đoạn CRISP-DM                | Việc chính                                                            | Mốc kiểm tra                                   |
| ----- | ------------ | ----------------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------ |
| W1    | 06–12/10    | **1. Business Understanding** | Tài liệu này; chốt phạm vi, metric, giả định                    | ✅**M0: đề cương GĐ1 được duyệt** |
| W2    | 13–19/10    | 2. Data Understanding               | Tải 8 file,`verify_dataset_facts.py`, profiling missing/zero/outlier | M1: xác nhận Q1–Q14 bằng số                 |
| W3    | 20–26/10    | 2 → 3                              | `bureau`+`bureau_balance`, `previous_application`: agg & QA       | M2: bảng agg ≥ 30 cột                         |
| W4    | 27/10–02/11 | 3. Data Preparation                 | POS / installments / credit_card: agg 3 tầng; gộp vào 1 ma trận     | ✅**M3: ΔAUC sơ bộ (G3)**               |
| W5    | 03–09/11    | 3                                   | Làm sạch, winsorize, WOE/IV, encoding, missing-indicator              | M4: ma trận 307 511 × ≤300                    |
| W6    | 10–16/11    | 4. Modelling                        | LogReg baseline (L1/L2, C search) + chuẩn hoá                         | M5: LogReg ≥ 0,72                               |
| W7    | 17–23/11    | 4                                   | RF: randomized search, OOB, permutation importance                      | M6: RF ≥ 0,76                                   |
| W8    | 24–30/11    | 3 + 5                               | Tối ưu pipeline (G5): parquet, chunked, benchmark                     | ✅**M7: pipeline ≤ 45 phút**             |
| W9    | 01–07/12    | 4                                   | Mất cân bằng: 5 phương án (a)–(e)                                | M8: bảng so sánh G4                            |
| W10   | 08–14/12    | 5. Evaluation                       | GroupKFold, CI bootstrap, decile/lift, calibration                      | ✅**M9: AUC ≥ 0,76–0,78 (G1) & AP (G2)** |
| W11   | 15–21/12    | 5                                   | SHAP + WOE, đường cong lợi nhuận, fairness, ablation hoàn chỉnh  | M10: draft chương 4–5                         |
| W12   | 22–28/12    | 6. Deployment                       | Bảng chính sách duyệt vay, báo cáo, slide, tái lập 1 lệnh      | ✅**M11: nghiêm thu**                     |

### 6.2. Sổ rủi ro dự án

| #   | Rủi ro                                                      | Xác suất         | Tác động                       | Giảm thiểu                                                                                                                                                |
| --- | ------------------------------------------------------------ | ------------------ | --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| RR1 | Máy cá nhân không đủ RAM cho 31,2M dòng               | Cao                | Chặn đứng G3, G5               | gộp theo`chunksize` + parquet + dtype float32; chuẩn bị Lending Club (2,2M dòng) làm kế hoạch B; dùng `pd.Categorical`                          |
| RR2 | Merge gây fan-out → mọi chỉ số sai mà không báo lỗi | **Cao**      | Toàn bộ                         | `assert len()==307_511` và `index.is_unique` ở **mỗi** bước merge (không đàm phán)                                                       |
| RR3 | Leakage từ bảng phụ (Q11) → hiệu năng ảo              | Trung bình        | Mất uy tín khi phản biện hỏi | Ghi log "cửa sổ thông tin" của từng biến; kiểm tra`DAYS_*<0`; test trên `application_test`                                                      |
| RR4 | Thời gian chạy RF trên 300 cột × 300k dòng quá lâu   | Cao                | Trễ W7                           | giới hạn ≤ 200 cột sau IV;`n_estimators=300`, `n_jobs=-1`, subsample 50% cho tìm kiếm; hoặc HistGradientBoosting làm proxy nhanh                |
| RR5 | Kết quả không đạt AUC 0,78                              | Trung bình        | G1 trượt                        | Thang dự phòng: 0,76 vẫn đạt ngưỡng "ĐẠT"; nếu <0,74 → xem lại leakage/grain trước khi đổi mô hình                                        |
| RR6 | Hội đồng hỏi "sao không dùng XGBoost?"                 | **Rất cao** | Mất điểm                       | đã trả lời trước ở RQ3: đề tài*tập trung* LogReg vs RF theo yêu cầu; thêm 1 hàng XGB/LGBM làm **tham chiếu trần** nếu còn trang |
| RR7 | Số liệu vĩ mô VN trong báo cáo lỗi thời khi bảo vệ | Trung bình        | Nhẹ                              | Ghi ngày truy cập + nguồn; chỉ dùng số của 2024–2025 đã công bố chính thức                                                                    |
| RR8 | SMOTE làm mất ý nghĩa xác suất (H4 đúng)             | Trung bình        | G4 lệch                          | hiệu chuẩn lại bằng isotonic/Platt**sau** resample; báo Brier bắt buộc                                                                         |

### 6.3. Tiêu chí nghiệm thu Giai đoạn 1 (bảng kiểm để hội đồng chấm ngay từ đề cương)

- [X] Nỗi đau kinh doanh được **định lượng** (8 pain points, mỗi cái có đại lượng đo được trong dữ liệu).
- [X] Có **hàm mục tiêu kinh tế** tường minh (`p* = r/(r+LGD)`), không chỉ "tăng độ chính xác".
- [X] 7 mục tiêu SMART, mỗi mục tiêu có baseline + ngưỡng đạt + ngưỡng xuất sắc + ngày kiểm tra.
- [X] 5 (+1) câu hỏi nghiên cứu, mỗi câu có **giả thuyết kiểm định được** và phép thử cụ thể.
- [X] Y được định nghĩa **vận hành** (ngưỡng DPD, back-in-cure, cửa sổ 12 tháng) và phân biệt với chậm trả/mất vốn.
- [X] X được lập bản đồ 12 khối → bảng nguồn → kỹ thuật → kỳ vọng dấu; 15 biến phái sinh có công thức.
- [X] Bài toán được phân loại **rõ ràng** là phân loại nhị phân có giám sát (kèm lập luận loại trừ hồi quy/phân cụm).
- [X] 14 bẫy dữ liệu đăng ký trước, mỗi cái có cách xử lý và hậu quả nếu bỏ qua.
- [X] Ma trận truy xuất mục tiêu ↔ câu hỏi ↔ biến ↔ chỉ tiêu ↔ deliverable.
- [X] Mọi con số **được dán nhãn [K]/[M]/[C]**; không có kết quả mô hình nào được trình bày như đã đạt.

---

## 7. VIỆC CẦN LÀM NGAY Ở GIAI ĐOẠN 2 (chuyển giao có kiểm soát)

Tám con số **phải** được đo để thay thế nhãn [C] trong tài liệu này; mỗi con số là một `print` trong `tools/verify_dataset_facts.py` (đã viết sẵn, chạy được ngay khi có file):

| #  | Đại lượng                                                          | Nếu khác giả định thì sửa ở đâu |
| -- | ---------------------------------------------------------------------- | ----------------------------------------- |
| V1 | `TARGET.mean()` — tỷ lệ vỡ nợ thực                             | 4.2; π trong mục 1.5; mọi baseline     |
| V2 | Số cột, dtype counts (numeric/categorical) của`application_train` | 4.4                                       |
| V3 | Tỷ trọng`DAYS_EMPLOYED > 0`                                        | Q1, nhóm biến thâm niên (X4)          |
| V4 | Missing-rate > 60% của khối X9/X6                                    | Q9, số cột đưa vào mô hình         |
| V5 | `SK_ID_BUREAU` trùng lặp; `CREDIT_CURRENCY` counts               | Q5, Q6                                    |
| V6 | Số dòng trung bình mỗi cấp: 1 hồ sơ → n khoản → m tháng     | G5 (RAM/time), Hình 1                    |
| V7 | Tỷ lệ`installments_payments` có >1 dòng/kỳ hạn                 | Q8, công thức F9/F10                    |
| V8 | AUC đơn biến của`EXT_SOURCE_1/2/3`                               | H1 (RQ1)                                  |

**Lệnh chạy** (khi đã có thư mục dữ liệu gốc `data/home-credit/`):

```bash
python3 tools/verify_dataset_facts.py --data data/home-credit
```

Kết quả mong đợi: một bảng text + `output/facts.json`, và `exit code ≠ 0` nếu bất biến grain (307 511) bị vi phạm — để Giai đoạn 3 không bao giờ xây trên một mẫu đã nứt.

---

## PHỤ LỤC A — Bảng kích thước dữ liệu thực tế [K]

| # | File                                         | Số dòng            | Số cột | Grain                           | Khoá nối lên               |
| - | -------------------------------------------- | -------------------- | -------- | ------------------------------- | ----------------------------- |
| 0 | `application_train.csv`                    | **307 511**    | 122      | 1 hồ sơ                       | — (gốc)                     |
| 0 | `application_test.csv`                     | 48 744               | 121      | 1 hồ sơ                       | —                            |
| 1 | `bureau.csv`                               | 1 716 428            | 17       | 1 khoản vay tại TCTD khác    | `SK_ID_CURR`                |
| 2 | `bureau_balance.csv`                       | 27 299 925           | 3        | 1 tháng của 1 khoản          | `SK_ID_BUREAU`              |
| 3 | `previous_application.csv`                 | 1 670 214            | 41       | 1 đơn xin vay cũ             | `SK_ID_CURR`                |
| 4 | `POS_CASH_balance.csv`                     | 10 001 358           | 10       | 1 tháng của 1 khoản POS/cash | `SK_ID_PREV`/`SK_ID_CURR` |
| 5 | `installments_payments.csv`                | 13 605 401           | 11       | 1 kỳ hạn/1 lần trả          | `SK_ID_PREV`/`SK_ID_CURR` |
| 6 | `credit_card_balance.csv`                  | 3 840 312            | 23       | 1 tháng của 1 thẻ            | `SK_ID_PREV`/`SK_ID_CURR` |
|   | **Σ 5 bảng phụ (đề bài nêu)**   | **30 833 713** |          |                                 |                               |
|   | **Σ 6 bảng (train+test+5 phụ)**     | **31 189 968** |          |                                 |                               |
|   | **Σ tất cả (thêm bureau_balance)** | **58 489 893** |          |                                 |                               |

Suy ra: trung bình mỗi hồ sơ gánh **≈100 dòng** ở 5 bảng phụ (installments 44,2 · POS 32,5 · bureau 5,6 · previous_application 5,4) **[K]** — con số đủ để giải thích với hội đồng *tại sao* "ghép nối đa cấp + tổng hợp" là kỹ thuật trung tâm chứ không phải bước phụ.

> **Đính chính so với đề bài:** (i) đề bài ghi "6 bảng phụ" nhưng liệt kê 5; `bureau_balance.csv` là bảng thứ 6 và là **bảng lớn nhất** (27,3M dòng) — được đưa vào phạm vi vì không thể dùng trạng thái quá hạn của bureau mà không đi qua nó; (ii) đề bài ghi "tổng quy mô vượt trên 10 triệu bản ghi" — con số chính xác là **31,19 triệu** (6 bảng) / **58,49 triệu** (7 bảng), tức đề bài đang **ước lượng thấp đi ~3 lần**; (iii) `application_train.csv` có **122** cột (121 biến + TARGET), không phải 121.

## PHỤ LỤC B — Thuật ngữ

| Thuật ngữ      | Nghĩa trong tài liệu                                                                                                       |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| PD               | Xác suất vỡ nợ — đại lượng thực sự cần ước lượng                                                              |
| LGD / EAD        | Tổn thất khi vỡ nợ / dư nợ tại thời điểm vỡ nợ;`EL = PD × LGD × EAD`                                          |
| Scorecard        | Mô hình xếp hạng điểm; ở đây là`f(X) → PD → điểm`                                                             |
| Scorecut-line    | Ngưỡng cắt duyệt/từ chối; đề tài xác định =`p* = 20%` theo chi phí                                             |
| KS               | max\|TPR − FPR\| — khoảng cách tách 2 lớp, chuẩn ngành scorecard                                                      |
| WOE / IV         | Weight of Evidence / Information Value — mã hoá biến & đo sức mạnh đơn biến, quen thuộc với kiểm toán mô hình |
| Lift / decile    | Gộp khách theo điểm thành 10 nhóm, so tỷ lệ vỡ nợ với trung bình danh mục                                        |
| Thin file        | Khách hàng hầu như không có lịch sử tín dụng → missing nhiều nhưng**không** phải rủi ro cao             |
| Reject inference | Kỹ thuật suy luận cho hồ sơ bị từ chối (không có nhãn) để sửa chệch chọn mẫu                                 |
| Back-in-cure     | Khoản quá hạn rồi trả đủ và được đưa về nhóm nợ cũ — làm nhãn "mềm" nếu không quản lý                |
| PSI              | Population Stability Index — đo trôi phân phối điểm giữa các kỳ                                                     |

## PHỤ LỤC C — Nguồn tham chiếu chính

1. Home Credit N.V. / Kaggle (2017), *Home Credit Default Risk* — tài liệu mô tả dữ liệu chính thức (HomeCredit_columns_description), số dòng mỗi bảng, metric ROC AUC của cuộc thi.
2. NHNN / Văn phòng Chính phủ, *Báo cáo tổng hợp việc thực hiện các nghị quyết của Quốc hội về giám sát chuyên đề lĩnh vực ngân hàng*: tăng trưởng tín dụng 2023 (+13,78%), 2024 (+15,09%), 30/9/2025 (+13,86% YTD), nợ xấu nội bảng cuối 10/2025 (1,64%, không gồm 5 nhà băng kiểm soát đặc biệt).
3. VietnamPlus, *Credit growth nears 18% in 2025* (họp báo NHNN 29/12/2025): dư nợ > 18,4 triệu tỷ đồng, +17,87%.
4. VnExpress / Báo Thanh tra, dẫn báo cáo tổng kết thi hành Nghị quyết 42: nợ xấu nội bảng toàn ngành 4,3% (1/2025), 4,75% (7/2024), 4,55% (cuối 2023), ~2% (cuối 2022); nợ xấu 27 ngân hàng niêm yết Q1/2025 > 266 nghìn tỷ đồng (+18,5%).
5. Thông tư 11/2021/TT-NHNN và các thông tư sửa đổi/gia hạn về phân loại nợ & trích lập dự phòng; Luật Các tổ chức tín dụng 2024; Nghị định 13/2023/NĐ-CP về bảo vệ dữ liệu cá nhân — *cần đối chiếu bản hợp nhất hiện hành tại thời điểm bảo vệ*.

**Tuyên bố về số liệu:** các con số mục 1.5 và Hình 2 là **kết quả của mô hình giải tích** trên phân phối giả lập với tham số công khai tại bảng 1.5 (không phải kết quả thực nghiệm trên dữ liệu vay, và không phải cam kết hiệu năng). Toàn bộ hiệu năng mô hình trong tài liệu này là **chỉ tiêu phải đạt**.
