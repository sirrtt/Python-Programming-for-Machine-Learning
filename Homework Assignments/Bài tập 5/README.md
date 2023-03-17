# Bài tập 5 - Linear Regression + đánh giá mô hình + Streamlit
Các bạn nộp bài tập hồi quy tuyến tính sử dụng 2 phương pháp Train/Test split và K-fold cross validation để đánh giá mô hình.

Sử dụng Streamlit để làm giao diện ứng dụng theo gợi ý trên lớp lý thuyết.

Yêu cầu bao gồm:
Thiết kế giao diện với Streamlit để có thể:
- Upload file csv (sau này có thể thay bằng tập dữ liệu khác dễ dàng).
- Hiển thị bảng dữ liệu với file đã upload
- Chọn lựa input feature (các cột dữ liệu đầu vào)
- Chọn lựa hệ số cho train/test split: Ví dụ 0.8 có nghĩa là 80% để train và 20% để test
- Chọn lựa hệ số K cho K-Fold cross validation: Ví dụ K =4
- Nút "Run" để tiến hành chạy và đánh giá thuật toán

Output sẽ là biểu đồ cột hiển thị các kết quả sử dụng độ đo MAE và MSE. Lưu ý: Train/Test split và K-Fold cross validation được thực hiện độc lập, chỉ chọn 1 trong hai phương pháp này.
