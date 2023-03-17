# Bài tập 7 - Classification với PCA để giảm số chiều
Sử dụng Wine dataset, kết hợp với streamlit:

- Bổ sung thêm option PCA, cho phép nhập số chiều sau khi giảm.

- Input feature X sau khi đã giảm chiều sẽ biến thành X'. Dùng X' để huấn luyện và dự đoán.

Lưu ý: Mô hình giảm số chiều được thực hiện trên tập train, thì sẽ giữ nguyên tham số để áp dụng trên tập test, chứ không fit lại trên tập test.

Tutorial: Nhấn https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_iris.html đường dẫn để mở nguồn.
