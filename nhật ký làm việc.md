# Nhật ký Làm việc Dự án

Tệp này tự động lưu vết các phiên làm việc giữa Nhóm sinh viên và Antigravity IDE.
- **Tên đề tài:** PRJ-01 - Dự đoán Rủi ro Vỡ nợ Tín dụng (Home Credit Default Risk)
- **Nhóm sinh viên thực hiện:**
  - Nguyễn Đỗ Tuấn Tú - MSSV: 2521001112
  - Huỳnh Đức Phương Thảo - MSSV: 2521001097

---

## [2026-10-05 15:15] - Khám Phá Cấu Trúc Sơ Bộ Ma Trận Đặc Trưng Hoàn Chỉnh (train_merged_final.parquet)
- **Môi trường:** Windows (PowerShell)
- **Mục tiêu phiên:** Thực hiện khám phá cấu trúc sơ bộ trên toàn bộ ma trận đặc trưng sau ghép nối [train_merged_final.parquet](file:///d:/University/Chương trình đào tạo/Năm 2026/HK3 - 2026/Lập trình Python cho khoa học dữ liệu/PRJ-01/train_merged_final.parquet): đo lường kích thước dòng/cột (`df.shape`), kiểm tra mức tiêu thụ bộ nhớ RAM và đĩa (`df.info()`), tính toán thống kê 5 con số cơ bản (`df.describe()`) trên các biến định lượng và lập bảng phân phối tần số cho các biến định tính cốt lõi.
- **Nội dung đã hoàn thành:**
  - Viết và hoàn thiện module phân tích [eda_structural_summary.py](file:///d:/University/Chương trình đào tạo/Năm 2026/HK3 - 2026/Lập trình Python cho khoa học dữ liệu/PRJ-01/source/eda_structural_summary.py).
  - Trích xuất toàn diện bức tranh cấu trúc của tập dữ liệu sau ghép nối: 307,511 dòng x 734 cột, dung lượng file đĩa 168.85 MB, tiêu thụ RAM 1,095.07 MB.
  - Phân tích chi tiết 718 biến định lượng và 16 biến định tính gốc từ hồ sơ đơn vay.
- **Tương tác Prompt & Quyết định kỹ thuật:**
  - *Prompt chiến lược đã thực thi:* "Khám phá cấu trúc sơ bộ: – Kích thước tập dữ liệu sau ghép nối (df.shape): Số lượng dòng và số lượng cột. – Kiểm tra mức tiêu thụ bộ nhớ (df.info(memory_usage='deep')). – Thống kê 5 con số cơ bản đối với biến định lượng (df.describe()) và phân phối tần số đối với biến định tính."
  - *Kinh nghiệm bóc tách code & Mutate/Break:*
    - Bóc tách phân phối lệch (Skewness & Outlier Trap): Các biến tài chính cốt lõi như `AMT_INCOME_TOTAL` (Max = 117 triệu so với Median = 147,150) và `BUREAU_AMT_CREDIT_SUM_MEAN` (Max = 198 triệu) có độ lệch phải cực lớn với các outlier cá biệt; trung vị (Median / 50%) phản ánh xu hướng trung tâm trung thực hơn trung bình (Mean).
    - Bóc tách mất cân bằng nhãn: Tỷ lệ `TARGET` đạt 8.07% (24,825 hồ sơ vỡ nợ trên tổng số 307,511 hồ sơ), xác lập chính xác tỷ lệ mất cân bằng ~1 : 11.4 cần kỹ thuật phân tầng `StratifiedKFold`.
    - Bóc tách tỷ lệ khuyết thiếu tự nhiên (Structural Missingness): Biến dư nợ thẻ `CC_AMT_BALANCE_MEAN` chỉ có 86,905 dòng có giá trị (~28.2%), phản ánh đúng thực tế chỉ một bộ phận khách hàng được cấp thẻ tín dụng; khi huấn luyện cần gán giá trị 0 hoặc tạo biến chỉ báo (Indicator).
  - *Quyết định kiến trúc:* Nhờ kỹ thuật downcast ép kiểu `float32` (677 cột) và `int32` (39 cột) thực hiện ở bước trước, ma trận 734 cột với >225 triệu ô dữ liệu chỉ chiếm ~1.07 GB RAM, hoàn toàn phù hợp để nạp trực tiếp vào bộ nhớ huấn luyện mô hình.
- **Tồn đọng & Bước tiếp theo:**
  - 1. Phân tích ma trận tương quan giữa các đặc trưng tổng hợp với biến mục tiêu `TARGET` để chọn lọc đặc trưng (Feature Selection).
  - 2. Xây dựng pipeline điền khuyết (Imputation) và mã hóa (Target Encoding / One-Hot) cho 16 biến định tính còn lại.
  - 3. Thiết lập baseline model với Logistic Regression và Random Forest theo thang đo ROC-AUC.

---

## [2026-10-05 15:03] - Xây Dựng Kịch Bản Trọn Gói Xuất File Parquet Hoàn Chỉnh (build_train_merged.py)
- **Môi trường:** Windows (PowerShell)
- **Mục tiêu phiên:** Xây dựng kịch bản Python thực thi trọn gói (End-to-End Execution) [build_train_merged.py](file:///d:/University/Chương trình đào tạo/Năm 2026/HK3 - 2026/Lập trình Python cho khoa học dữ liệu/PRJ-01/source/build_train_merged.py) tự động hóa toàn bộ quy trình: đọc `application_train.csv`, tuần tự aggregate, làm phẳng và merge an toàn cả 5 bảng phụ (`bureau`, `previous_application`, `POS_CASH`, `installments`, `credit_card`), bảo toàn đúng 307,511 dòng và xuất thành phẩm siêu nén `train_merged_final.parquet`.
- **Nội dung đã hoàn thành:**
  - Viết và hoàn thiện file kịch bản trọn gói [build_train_merged.py](file:///d:/University/Chương trình đào tạo/Năm 2026/HK3 - 2026/Lập trình Python cho khoa học dữ liệu/PRJ-01/source/build_train_merged.py) trong package `source`.
  - Cập nhật [__init__.py](file:///d:/University/Chương trình đào tạo/Năm 2026/HK3 - 2026/Lập trình Python cho khoa học dữ liệu/PRJ-01/source/__init__.py) xuất hàm `run_build_train_merged`.
  - Thiết kế cơ chế xử lý luân phiên từng file một (Single-table streaming execution): Đọc -> Aggregate -> Làm phẳng -> Merge -> Downcast -> Giải phóng RAM tức thì, bảo đảm không bị tràn RAM dù chạy toàn bộ 31 triệu dòng dữ liệu trên máy tính cá nhân.
- **Tương tác Prompt & Quyết định kỹ thuật:**
  - *Prompt chiến lược đã thực thi:* "giúp tôi làm luôn nha (cách 1)" (Lưu bảng hoàn chỉnh train_merged_final.parquet).
  - *Kinh nghiệm bóc tách code & Mutate/Break:*
    - Bóc tách quản lý RAM theo chu kỳ vòng lặp: Với các bảng lớn như `installments_payments` (13.6 triệu dòng) và `POS_CASH_balance` (10 triệu dòng), nếu nạp đồng thời 5 bảng vào RAM sẽ cần trên 16GB RAM. Giải pháp: Kịch bản chỉ nạp và giữ duy nhất bảng `application_train` trong suốt vòng đời, mỗi bảng phụ được nạp vào, aggregate xong thì `del` và `gc.collect()` ngay trước khi mở bảng tiếp theo.
    - Tích hợp bước kiểm chứng tự động (Verification Check): Ngay sau khi nén file `train_merged_final.parquet`, kịch bản tự động đọc ngược lại 5 dòng để kiểm tra tính toàn vẹn của file trên ổ đĩa.
  - *Quyết định kiến trúc:* Lưu định dạng `.parquet` bằng engine `pyarrow` với chuẩn nén snappy, giúp giảm dung lượng từ gần 1.5 GB CSV xuống còn ~150 - 250 MB và tăng tốc độ đọc lên gấp 10 lần.
- **Tồn đọng & Bước tiếp theo:**
  - 1. Chạy thực thi `python source/build_train_merged.py` trên terminal để tạo file `train_merged_final.parquet`.
  - 2. Thực hiện phân tích dữ liệu khuyết thiếu trên tập dữ liệu hoàn chỉnh và xây dựng pipeline baseline model (Logistic Regression & Random Forest).

---

## [2026-10-05 14:44] - Xây Dựng Pipeline Ghép Nối Tuần Tự 5 Bảng Phụ (sequential_merge_pipeline) Với Chốt Chặn Toàn Vẹn
- **Môi trường:** Windows (PowerShell)
- **Mục tiêu phiên:** Xây dựng mã nguồn Python hoàn chỉnh thực thi pipeline ghép nối tuần tự 5 bảng đặc trưng phụ (`bureau_agg`, `prev_agg`, `pos_agg`, `install_agg`, `credit_agg`) vào bảng chính `application_train`, kiểm soát triệt để hiện tượng nhân bản dòng (fan-out với assert `len(df_main) == 307511`), quét sạch xung đột cột `_x, _y`, tự động downcast các cột mới thêm và giải phóng RAM tức thì sau mỗi bước.
- **Nội dung đã hoàn thành:**
  - Viết và hoàn thiện hàm [merge_pipeline.py](file:///d:/University/Chương trình đào tạo/Năm 2026/HK3 - 2026/Lập trình Python cho khoa học dữ liệu/PRJ-01/source/merge_pipeline.py) chứa pipeline tuần tự `sequential_merge_pipeline`.
  - Cập nhật [__init__.py](file:///d:/University/Chương trình đào tạo/Năm 2026/HK3 - 2026/Lập trình Python cho khoa học dữ liệu/PRJ-01/source/__init__.py) xuất hàm `sequential_merge_pipeline`.
  - Thiết kế cơ chế theo dõi RAM động và tự động downcast `float64 -> float32`, `int64 -> int32` cho đúng các cột đặc trưng mới phát sinh sau mỗi lần merge.
- **Tương tác Prompt & Quyết định kỹ thuật:**
  - *Prompt chiến lược đã thực thi:* "Tôi cần nối tuần tự 5 bảng phụ đã aggregate về cấp độ SK_ID_CURR: 1. bureau_agg (BUREAU_) 2. prev_agg (PREV_) 3. pos_agg (POS_) 4. install_agg (INSTAL_) 5. credit_agg (CC_). Hãy viết mã nguồn Python hoàn chỉnh thực thi pipeline ghép nối với các yêu cầu kỹ thuật: 1. Cơ chế nối tuần tự qua vòng lặp, assert len == 307511, del & gc.collect(), in tiến độ... 2. Kiểm soát xung đột cột _x, _y. 3. Downcasting dữ liệu: float64->float32, int64->int32 cho cột mới..."
  - *Kinh nghiệm bóc tách code & Mutate/Break:*
    - Bóc tách tự động xác định cột mới (Selective Downcasting): Bằng cách lấy hiệu tập hợp cột `new_cols = [c for c in df_main.columns if c not in cols_before]`, hàm chỉ ép kiểu đúng các cột vừa nhập từ bảng phụ hiện tại mà không phải quét toàn bộ bảng hàng trăm cột, tối ưu tốc độ thực thi đáng kể.
    - Bóc tách chốt chặn dòng lặp (Per-Step Assertion): Đặt lệnh assert `len(df_main) == 307511` ngay trong từng vòng lặp giúp phát hiện chính xác bảng phụ nào bị lỗi aggregate (nếu có) thay vì để merge hết 5 bảng rồi mới kiểm tra, giúp cô lập lỗi tức thì.
  - *Quyết định kiến trúc:* Cung cấp cấu trúc danh sách tuple linh hoạt `[('bureau', bureau_agg), ...]`, kết hợp dọn rác bộ nhớ `del df_sec; gc.collect()` sau mỗi vòng lặp giúp dung lượng RAM luôn ổn định trên máy tính cá nhân.
- **Tồn đọng & Bước tiếp theo:**
  - 1. Kiểm định ma trận đặc trưng cuối cùng (Missing value ratio, infinite values).
  - 2. Áp dụng kỹ thuật tiền xử lý dữ liệu tài chính (Imputation theo domain knowledge) và chuẩn bị tập Cross-Validation (Stratified K-Fold).

---

## [2026-10-05 14:38] - Xây Dựng Hàm Ghép Nối Dữ Liệu An Toàn (safe_merge) Chống Bùng Nổ Dòng & Xung Đột Cột
- **Môi trường:** Windows (PowerShell)
- **Mục tiêu phiên:** Xây dựng hàm Python `safe_merge` trong thư mục `source` thực hiện `LEFT JOIN` giữa bảng chính `application_train` (307,511 dòng) và các bảng đặc trưng phụ đã được tổng hợp, tích hợp 3 tầng chốt chặn nghiêm ngặt: kiểm định toàn vẹn số dòng (`len(df_merged) == len(application_train)`), phát hiện xung đột cột (`_x`, `_y`), và giải phóng RAM tức thì.
- **Nội dung đã hoàn thành:**
  - Viết và hoàn thiện hàm [safe_merge.py](file:///d:/University/Chương trình đào tạo/Năm 2026/HK3 - 2026/Lập trình Python cho khoa học dữ liệu/PRJ-01/source/safe_merge.py) trong package `source`.
  - Cập nhật [__init__.py](file:///d:/University/Chương trình đào tạo/Năm 2026/HK3 - 2026/Lập trình Python cho khoa học dữ liệu/PRJ-01/source/__init__.py) xuất hàm `safe_merge`.
  - Kiểm thử logic thành công: Đo lường dung lượng RAM giải phóng được từ bảng phụ, chặn đứng nguy cơ Row Explosion và cột trùng lặp.
- **Tương tác Prompt & Quyết định kỹ thuật:**
  - *Prompt chiến lược đã thực thi:* "Tôi đã có bảng chính application_train (shape gốc: 307,511 dòng) và bảng đặc trưng phụ đã được làm sạch, aggregate ở bước 2... 1. Thao tác Merge: pd.merge(application_train, df_aggregated, on='SK_ID_CURR', how='left'). 2. Kiểm tra tính toàn vẹn: len(df_merged) == len(application_train); không phát sinh _x, _y. 3. Quản lý bộ nhớ: del và gc.collect()."
  - *Kinh nghiệm bóc tách code & Mutate/Break:*
    - Bóc tách kiểm soát bùng nổ dòng (Row Explosion Defense): Nếu bảng phụ chưa được gom nhóm triệt để về 1:1, lệnh `LEFT JOIN` sẽ tự động nhân bản dòng ở các bản ghi trùng `SK_ID_CURR`, làm lệch hoàn toàn phân phối của `TARGET` và gây sai lệch mô hình. Câu lệnh `assert len(df_merged) == len(application_train)` đóng vai trò như chốt an toàn bắt buộc trước khi dữ liệu được chuyển sang bước kế tiếp.
    - Bóc tách kiểm soát xung đột cột (Suffix Collision Trap): Kiểm tra danh sách cột có đuôi `_x`, `_y`. Sự xuất hiện của các đuôi này chứng minh bước đặt prefix trước đó bị bỏ sót hoặc có cột ngoài khóa bị trùng tên mà không được tiền tố hóa.
  - *Quyết định kiến trúc:* Thiết lập cơ chế đo lường dung lượng RAM chính xác trước và sau khi `del secondary_df` kết hợp `gc.collect()`, giúp pipeline có thể chạy tuần tự qua cả 5 bảng phụ mà không bị nghẽn RAM máy tính cá nhân.
- **Tồn đọng & Bước tiếp theo:**
  - 1. Ghép nối tuần tự cả 5 bảng phụ (`bureau`, `previous_application`, `POS_CASH`, `installments`, `credit_card`) vào `application_train`.
  - 2. Kiểm tra tỷ lệ dữ liệu khuyết thiếu sau ghép nối và gắn nhãn theo kiến thức kinh tế (ví dụ: gán 0 cho các chỉ số dư nợ của khách hàng không có lịch sử).

---

## [2026-10-05 14:26] - Xây Dựng Hàm Chuẩn Hóa Schema (flatten_aggregated_df): Làm Phẳng MultiIndex & Phục Hồi Khóa Chính
- **Môi trường:** Windows (PowerShell)
- **Mục tiêu phiên:** Xây dựng giải pháp chuẩn hóa schema cho DataFrame sau aggregate: làm phẳng MultiIndex dạng `(tên_cột_gốc, phép_toán)` thành chuỗi phẳng viết hoa kèm tiền tố nguồn `[PREFIX]_[TÊN_CỘT_GỐC]_[PHÉP_TOÁN]`, phục hồi `SK_ID_CURR` về cột dữ liệu thông thường không dính tiền tố, và thực hiện Validation Check kiểm tra tính toàn vẹn 1:1.
- **Nội dung đã hoàn thành:**
  - Viết và hoàn thiện hàm [flatten_aggregated_df.py](file:///d:/University/Chương trình đào tạo/Năm 2026/HK3 - 2026/Lập trình Python cho khoa học dữ liệu/PRJ-01/source/flatten_aggregated_df.py) trong package `source`.
  - Cập nhật [__init__.py](file:///d:/University/Chương trình đào tạo/Năm 2026/HK3 - 2026/Lập trình Python cho khoa học dữ liệu/PRJ-01/source/__init__.py) xuất hàm `flatten_aggregated_df`.
  - Kiểm thử logic thành công: Đổi tên cột trước khi `reset_index()` giúp `SK_ID_CURR` được bảo toàn nguyên vẹn 100% không bị dính tiền tố; câu lệnh assert xác thực không có trùng lặp ID.
- **Tương tác Prompt & Quyết định kỹ thuật:**
  - *Prompt chiến lược đã thực thi:* "Tiếp tục từ kết quả DataFrame đã aggregate theo SK_ID_CURR ở bước 1... 1. Làm phẳng MultiIndex: [PREFIX]_[TÊN_CỘT_GỐC]_[PHÉP_TOÁN]... 2. Phục hồi khóa chính: SK_ID_CURR từ index trở lại cột thông thường không dính prefix/suffix... 3. Validation Check: assert df['SK_ID_CURR'].duplicated().sum() == 0..."
  - *Kinh nghiệm bóc tách code & Mutate/Break:*
    - Bóc tách thứ tự thực thi (Order of Execution Trap): Nếu gọi `df.reset_index()` trước rồi mới làm phẳng `df.columns`, cột `SK_ID_CURR` sẽ vô tình bị gán tiền tố thành `BUREAU_SK_ID_CURR_`, gây lỗi khi ghép nối `pd.merge(..., on='SK_ID_CURR')`. Giải pháp: Giữ `SK_ID_CURR` ở index trong lúc đổi tên toàn bộ MultiIndex columns, sau đó mới gọi `reset_index()` để khóa chính xuất hiện ở vị trí cột đầu tiên hoàn toàn sạch sẽ.
    - Bóc tách kiểm định toàn vẹn: Dùng lệnh assert `df['SK_ID_CURR'].duplicated().sum() == 0` chặn đứng mọi nguy cơ dữ liệu phát sinh trùng lặp trước khi bước vào pipeline huấn luyện.
  - *Quyết định kiến trúc:* Đóng gói thành hàm độc lập tái sử dụng trong `source/flatten_aggregated_df.py` có khả năng tự động xử lý đuôi gạch dưới của prefix (vd: nhận cả `'BUREAU'` hoặc `'BUREAU_'`) và in báo cáo nghiệm thu rõ ràng.
- **Tồn đọng & Bước tiếp theo:**
  - 1. Thực hiện `LEFT JOIN` các bảng phụ đã làm phẳng vào `application_train` / `application_test`.
  - 2. Kiểm tra tỷ lệ dữ liệu khuyết thiếu sau ghép nối và gắn nhãn theo kiến thức kinh tế (ví dụ: gán 0 cho các chỉ số dư nợ của khách hàng không có lịch sử).

---

## [2026-10-05 13:20] - Xây Dựng Hàm Tổng Hợp Bảng Phụ (aggregate_secondary_table) Về Cấp Độ Khách Hàng SK_ID_CURR
- **Môi trường:** Windows (PowerShell)
- **Mục tiêu phiên:** Xây dựng hàm Python `aggregate_secondary_table` trong thư mục `source` nhận vào DataFrame của 5 bảng phụ (`bureau`, `previous_application`, `POS_CASH_balance`, `installments_payments`, `credit_card_balance`) để tổng hợp (aggregate) về cấp độ khách hàng `SK_ID_CURR` (đảm bảo quan hệ sau aggregate là 1 : 1 với bảng chính `application_train`), kiểm soát RAM bằng downcasting và giữ nguyên MultiIndex phục vụ làm phẳng tên cột.
- **Nội dung đã hoàn thành:**
  - Viết và hoàn thiện hàm [aggregate_secondary_table.py](file:///d:/University/Chương trình đào tạo/Năm 2026/HK3 - 2026/Lập trình Python cho khoa học dữ liệu/PRJ-01/source/aggregate_secondary_table.py) trong package `source`.
  - Cập nhật [__init__.py](file:///d:/University/Chương trình đào tạo/Năm 2026/HK3 - 2026/Lập trình Python cho khoa học dữ liệu/PRJ-01/source/__init__.py) xuất hàm `aggregate_secondary_table`.
  - Thử nghiệm hiệu năng và kiểm chứng thực tế: Chạy thành công trên bảng lớn nhất `installments_payments` (13,6 triệu dòng, downcast trong 0,53s, aggregate chỉ mất 2,49s) và `previous_application` (1,67 triệu dòng, aggregate 362 features trong 16,5s).
- **Tương tác Prompt & Quyết định kỹ thuật:**
  - *Prompt chiến lược đã thực thi:* "Bảng chính là application_train (khóa SK_ID_CURR). Hãy viết một hàm nhận vào DataFrame phụ [5 bảng phụ còn lại] để aggregate về cấp độ SK_ID_CURR (đảm bảo quan hệ sau aggregate là 1 : 1 với bảng chính). Yêu cầu kỹ thuật bắt buộc: 1. Tách loại biến (loại trừ SK_ID_PREV, SK_ID_BUREAU; OHE biến phân loại rồi aggregate mean, sum; biến định lượng agg min, max, mean, sum hoặc count). 2. Kiểm soát tài nguyên: downcast float64->float32, int64->int32. 3. Output: DataFrame gom nhóm theo SK_ID_CURR chưa reset index."
  - *Kinh nghiệm bóc tách code & Mutate/Break:*
    - Bóc tách tách rời 2 luồng tính toán (Decoupled Aggregation): Thay vì ghép nối OHE dummies với toàn bộ bảng 10-13 triệu dòng rồi mới `groupby` (dễ gây nghẽn RAM nghiêm trọng), hàm xử lý độc lập luồng định lượng `num_agg` và luồng phân loại `cat_agg` trên cấp độ `SK_ID_CURR`, sau đó dọn `gc.collect()` rồi ghép nối ở cấp độ 338k dòng.
    - Bóc tách kiểm soát khóa định danh: Tự động phát hiện và loại bỏ triệt để các khóa con cấp giao dịch (`SK_ID_PREV`, `SK_ID_BUREAU`) khỏi tính toán thống kê (tránh bẫy tính `mean()` trên mã ID vô nghĩa), chỉ giữ duy nhất `SK_ID_CURR` làm Group Key.
    - Khắc phục lỗi mã hóa Unicode Windows `cp1258` khi print log tiếng Việt bằng cấu hình `sys.stdout.reconfigure(encoding='utf-8')`.
  - *Quyết định kiến trúc:* Thiết lập cấu trúc đầu ra giữ nguyên `index = SK_ID_CURR` và `columns = pd.MultiIndex` (Level 0: tên biến, Level 1: phép thống kê), không vội `reset_index` để người dùng linh hoạt đặt tiền tố (prefix) theo tên bảng và làm phẳng cột ở bước pipeline kế tiếp mà không lo xung đột tên cột.
- **Tồn đọng & Bước tiếp theo:**
  - 1. Xây dựng hàm làm phẳng tên cột MultiIndex và gắn tiền tố (table prefix, vd: `PREV_`, `BUR_`) để tránh xung đột cột khi ghép nối.
  - 2. Thực hiện `LEFT JOIN` tuần tự các bảng phụ đã aggregate vào `application_train` / `application_test`.
  - 3. Kiểm định tỷ lệ dữ liệu khuyết sau ghép nối và lựa chọn phương án xử lý dữ liệu khuyết thiếu phù hợp theo bản chất nghiệp vụ (Thin-file / First-time borrower).

---

## [2026-10-05 11:00] - Xây Dựng Hàm Phân Tích Schema Ghép Nối (inspect_schema_for_merge) Cho 6 DataFrames
- **Môi trường:** Windows (PowerShell)
- **Mục tiêu phiên:** Xây dựng hàm Python `inspect_schema_for_merge` trong thư mục `source` nhằm tự động duyệt qua 6 DataFrames của bài toán Home Credit Default Risk, trích xuất schema (dòng, cột, dtype, unique, null), xác định các cặp khóa ghép nối (Join Keys) với bảng chính `application_train`, phát hiện xung đột trùng tên cột và xuất báo cáo text ngắn gọn.
- **Nội dung đã hoàn thành:**
  - Khởi tạo package mã nguồn `source` với [__init__.py](file:///d:/University/Chương trình đào tạo/Năm 2026/HK3 - 2026/Lập trình Python cho khoa học dữ liệu/PRJ-01/source/__init__.py).
  - Viết và hoàn thiện hàm [inspect_schema_for_merge](file:///d:/University/Chương trình đào tạo/Năm 2026/HK3 - 2026/Lập trình Python cho khoa học dữ liệu/PRJ-01/source/inspect_schema_for_merge.py) hỗ trợ linh hoạt 2 chế độ: nhận Dict DataFrame đã load trong RAM hoặc tự động đọc tuần tự từng tệp CSV kết hợp dọn rác `gc.collect()` chống tràn RAM.
  - Chạy thực nghiệm thành công trên toàn bộ 6 bảng dữ liệu thực tế (>31 triệu dòng), tự động xuất và lưu trữ kết quả phân tích đầy đủ tại tệp [schema_report.txt](file:///d:/University/Chương trình đào tạo/Năm 2026/HK3 - 2026/Lập trình Python cho khoa học dữ liệu/PRJ-01/source/schema_report.txt).
- **Tương tác Prompt & Quyết định kỹ thuật:**
  - *Prompt chiến lược đã thực thi:* "Tôi có 6 DataFrames: tại folder Dataset - Home Credit Default Risk (application_train.csv là df chính). Hãy viết cho tôi 1 hàm Python tên inspect_schema_for_merge trong folder 'source' để: 1. Duyệt qua từng DataFrame và in ra... 2. Chỉ ra các cặp Khóa ghép nối... 3. Tự động kiểm tra trùng tên cột... 4. Xuất kết quả dưới dạng text ngắn gọn..."
  - *Kinh nghiệm bóc tách code & Mutate/Break:*
    - Bóc tách bản số (Cardinality): Khóa `SK_ID_CURR` là duy nhất ở `application_train` (307,511 unique = 100%) nhưng lặp lại ở cả 5 bảng phụ (`bureau` 305,811 unique; `previous_application` 338,857 unique...) -> Xác lập quan hệ chuẩn `1 : N`.
    - Bẫy lỗi Bùng nổ Dòng (Row Explosion Trap): Nếu dùng trực tiếp `pd.merge(..., on='SK_ID_CURR')`, dữ liệu sẽ bị nhân đôi nhân ba số dòng do quan hệ 1:N (đặc biệt `installments_payments` có 13.6 triệu dòng, `POS_CASH_balance` có 10 triệu dòng), làm RAM tràn vỡ (OOM) và sai lệch biến mục tiêu `TARGET`.
    - Bẫy lỗi Xung đột Cột (Column Collisions): Phát hiện `previous_application` trùng 7 cột ngoài khóa (`AMT_CREDIT`, `AMT_ANNUITY`, `AMT_GOODS_PRICE`, `HOUR_APPR_PROCESS_START`, `NAME_CONTRACT_TYPE`, `NAME_TYPE_SUITE`, `WEEKDAY_APPR_PROCESS_START`), `bureau` trùng 1 cột (`AMT_ANNUITY`). Nếu merge không kiểm soát sẽ sinh đuôi `_x`, `_y` gây nhiễu ma trận đặc trưng.
    - Bẫy lỗi Unicode Console Windows: Khắc phục lỗi mã hóa ký tự tiếng Việt `cp1258` trên PowerShell bằng cách cấu hình `sys.stdout.reconfigure(encoding='utf-8')` kết hợp cơ chế fallback mã hóa an toàn.
  - *Quyết định kiến trúc:* Thiết kế hàm có tính tùy biến cao (chấp nhận cả Dict DataFrames hoặc Directory Path), tối ưu hóa bộ nhớ với cơ chế đọc đơn dòng / đơn file và dọn RAM tức thì, xuất báo cáo chuẩn văn bản sẵn sàng sao chép trao đổi với AI/cộng sự.
- **Tồn đọng & Bước tiếp theo:**
  - 1. Xây dựng pipeline Feature Engineering tổng hợp (Aggregation: min, max, mean, sum, std) trên từng bảng phụ theo `SK_ID_CURR` và gắn tiền tố (prefix) tránh xung đột cột.
  - 2. Thực hiện Left Join an toàn các bảng phụ đã tổng hợp vào `application_train`.
  - 3. Kiểm định tỷ lệ dữ liệu khuyết thiếu sau ghép nối và phân tích tương quan đặc trưng với nhãn rủi ro `TARGET`.

---

## [2026-10-05 06:00] - Chuẩn Hóa Hệ Thống Lưu Vết PRJ-01 Theo Mẫu Bắt Buộc Mục 8.2 & Chuẩn P-D-M-V
- **Môi trường:** Windows (PowerShell)
- **Mục tiêu phiên:** Chuẩn hóa toàn diện kỹ năng quản lý nhật ký phiên làm việc cho PRJ-01 (`nhat-ky-prj01`) theo đúng mẫu cấu trúc bắt buộc tại Mục 8.2 của đề bài, cập nhật thông tin 2 thành viên nhóm sinh viên và tích hợp 4 trụ cột P-D-M-V (Prompt & Produce, Deconstruct & Read, Mutate & Break, Validate & Refine).
- **Nội dung đã hoàn thành:**
  - Cập nhật toàn diện cấu trúc kỹ năng [SKILL.md](file:///d:/University/Chương trình đào tạo/Năm 2026/HK3 - 2026/Lập trình Python cho khoa học dữ liệu/.agents/skills/nhật ký làm việc prj-01/SKILL.md) đồng bộ 100% với mẫu cấu trúc chuẩn Mục 8.2.
  - Chuẩn hóa lại tệp nhật ký dự án [nhật ký làm việc.md](file:///d:/University/Chương trình đào tạo/Năm 2026/HK3 - 2026/Lập trình Python cho khoa học dữ liệu/PRJ-01/nhật ký làm việc.md) theo đúng mẫu bắt buộc: Tên đề tài, Nhóm sinh viên thực hiện (Tuấn Tú & Phương Thảo), cùng 5 mục gạch đầu dòng cố định cho mỗi phiên.
- **Tương tác Prompt & Quyết định kỹ thuật:**
  - *Prompt chiến lược đã thực thi:* "tiếp tục cập nhật skill nhật ký làm việc cho prj01 theo cấu trúc trên, nhóm sinh viên có: Nguyễn Đỗ Tuấn Tú 2521001112, Huỳnh Đức Phương Thảo 2521001097" kèm ảnh chụp mẫu cấu trúc chuẩn Mục 8.2.
  - *Kinh nghiệm bóc tách code & Mutate/Break:*
    - Bóc tách cấu trúc logic các câu lệnh then chốt: `pd.merge()` (cơ chế ghép bảng dữ liệu tín dụng, kiểm soát Row Explosion), `StratifiedKFold` (bảo toàn tỷ lệ nhãn lệch `TARGET` ~8% vỡ nợ, chống Data Leakage), `roc_auc_score` (metric diện tích dưới đường cong ROC tối ưu cho dữ liệu mất cân bằng nghiêm trọng).
    - Thiết lập khung theo dõi thử nghiệm biến dị tham số: thay đổi `class_weight='balanced'`, điều chỉnh `max_depth` của Random Forest/LightGBM để quan sát và lý giải sự trồi sụt của điểm AUC.
  - *Quyết định kiến trúc:* Chuẩn hóa 100% theo mẫu quy định của môn học (Mục 8.2), tích hợp hướng dẫn kiểm chứng mô hình theo nguyên lý tài chính thực tế (DTI, nợ quá hạn Bureau, thâm niên việc làm) và tối ưu hóa bộ nhớ (`reduce_mem_usage`).
- **Tồn đọng & Bước tiếp theo:**
  - 1. Bắt đầu phân tích khám phá dữ liệu (EDA) trên tập [application_train.csv](file:///d:/University/Chương trình đào tạo/Năm 2026/HK3 - 2026/Lập trình Python cho khoa học dữ liệu/PRJ-01/Dataset - Home Credit Default Risk/application_train.csv) theo chuẩn P-D-M-V.
  - 2. Thực hiện kỹ thuật đặc trưng ban đầu (Feature Engineering) và xây dựng Pipeline baseline phân loại rủi ro tín dụng.

---

## [2026-10-03 19:25] - Khởi Tạo Môi Trường Dự Án & Xây Dựng Bộ Kỹ Năng Quản Lý Nhật Ký PRJ-01
- **Môi trường:** Windows (PowerShell)
- **Mục tiêu phiên:** Thiết lập môi trường dự án PRJ-01 (Home Credit Default Risk), xây dựng và cấu hình bộ kỹ năng (Skill) Antigravity IDE chuyên biệt để quản lý nhật ký phiên làm việc độc lập, đồng thời chuẩn hóa cấu trúc tệp nhật ký theo đúng quy định đề bài.
- **Nội dung đã hoàn thành:**
  - Khảo sát cấu trúc thư mục dữ liệu dự án [PRJ-01](file:///d:/University/Chương trình đào tạo/Năm 2026/HK3 - 2026/Lập trình Python cho khoa học dữ liệu/PRJ-01) bao gồm các bảng: [application_train.csv](file:///d:/University/Chương trình đào tạo/Năm 2026/HK3 - 2026/Lập trình Python cho khoa học dữ liệu/PRJ-01/Dataset - Home Credit Default Risk/application_train.csv), [bureau.csv](file:///d:/University/Chương trình đào tạo/Năm 2026/HK3 - 2026/Lập trình Python cho khoa học dữ liệu/PRJ-01/Dataset - Home Credit Default Risk/bureau.csv), [previous_application.csv](file:///d:/University/Chương trình đào tạo/Năm 2026/HK3 - 2026/Lập trình Python cho khoa học dữ liệu/PRJ-01/Dataset - Home Credit Default Risk/previous_application.csv)...
  - Tạo mới và cấu hình bộ kỹ năng [SKILL.md](file:///d:/University/Chương trình đào tạo/Năm 2026/HK3 - 2026/Lập trình Python cho khoa học dữ liệu/.agents/skills/nhật ký làm việc prj-01/SKILL.md) (`nhat-ky-prj01`) dành riêng cho dự án PRJ-01.
  - Chuẩn hóa vị trí và tên tệp nhật ký thành [nhật ký làm việc.md](file:///d:/University/Chương trình đào tạo/Năm 2026/HK3 - 2026/Lập trình Python cho khoa học dữ liệu/PRJ-01/nhật ký làm việc.md) theo quy định chính thức.
- **Tương tác Prompt & Quyết định kỹ thuật:**
  - *Prompt chiến lược đã thực thi:* "Tạo một bộ skill ghi lại Nhật ký làm việc tương tự với file SKILL.md đã tạo nhưng được dùng riêng cho phiên làm việc của PRJ-01. Xác nhận công việc trước khi tôi cho thực thi."
  - *Kinh nghiệm bóc tách code & Mutate/Break:* Phát hiện bẫy lỗi ghi đè vào file nhật ký chung của môn học (`00_nhật ký làm việc.md`), thiết lập cơ chế cô lập ngữ cảnh riêng cho từng dự án.
  - *Quyết định kiến trúc:* Tách biệt tệp nhật ký của PRJ-01 khỏi nhật ký chung của môn học ([00_nhật ký làm việc.md](file:///d:/University/Chương trình đào tạo/Năm 2026/HK3 - 2026/Lập trình Python cho khoa học dữ liệu/00_nhật ký làm việc.md)) để tránh tràn ngữ cảnh và đảm bảo tính độc lập khi nộp đồ án.
- **Tồn đọng & Bước tiếp theo:**
  - 1. Kiểm tra tỷ lệ mất cân bằng nhãn mục tiêu (`TARGET`) trên tập [application_train.csv](file:///d:/University/Chương trình đào tạo/Năm 2026/HK3 - 2026/Lập trình Python cho khoa học dữ liệu/PRJ-01/Dataset - Home Credit Default Risk/application_train.csv).
  - 2. Thống kê tỷ lệ khuyết thiếu (Missing Values) và trực quan hóa phân phối các đặc trưng nhân khẩu học/tài chính.

---
