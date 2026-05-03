import csv
import re
import random
import math

# 1. Chuẩn hóa văn bản: Chuyển chữ thường, bỏ dấu câu, tách từ
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text) # Chỉ giữ lại chữ cái và số
    return text.split()

# 2. Đọc file dữ liệu
def load_data(file_path):
    dataset = []
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader) # Bỏ qua dòng tiêu đề (header)
        for row in reader:
            if len(row) >= 2:
                # Cần điều chỉnh index tùy theo cấu trúc file của bạn
                label = row[0].strip().lower() # Giả sử cột 0 là nhãn (spam/ham)
                text = row[1]                  # Giả sử cột 1 là nội dung email
                if label in ['spam', 'ham']:
                    dataset.append((clean_text(text), label))
    return dataset

# 3. Chia dữ liệu thành tập Train (để huấn luyện) và Test (để đánh giá)
def split_data(dataset, split_ratio=0.8):
    random.shuffle(dataset) # Xáo trộn ngẫu nhiên dữ liệu
    train_size = int(len(dataset) * split_ratio)
    train_set = dataset[:train_size]
    test_set = dataset[train_size:]
    return train_set, test_set

# 4. Xây dựng mô hình tần suất từ vựng từ tập huấn luyện
def build_frequency_model(train_set):
    vocab = set()
    word_freq = {'spam': {}, 'ham': {}}
    doc_count = {'spam': 0, 'ham': 0}

    for tokens, label in train_set:
        doc_count[label] += 1
        for word in tokens:
            vocab.add(word)
            word_freq[label][word] = word_freq[label].get(word, 0) + 1

    return vocab, word_freq, doc_count

# 5. Huấn luyện mô hình Naive Bayes
def train_naive_bayes(vocab, word_freq, class_count):
    total_docs = sum(class_count.values())
    log_priors = {}
    for class_label in class_count:
        log_priors[class_label] = math.log(class_count[class_label] / total_docs)

    total_words = {}
    for class_label in word_freq:
        total_words[class_label] = sum(word_freq[class_label].values())

    vocab_size = len(vocab)
    log_likelihoods = {'spam': {}, 'ham': {}}
    for word in vocab:
        for class_label in ['spam', 'ham']:
            count = word_freq[class_label].get(word, 0)
            log_likelihoods[class_label][word] = math.log((count + 1) / (total_words[class_label] + vocab_size))
            # Sử dụng Laplace smoothing (count + 1) để tránh log(0) khi từ không xuất hiện trong lớp đó

    return log_priors, log_likelihoods

# 6. Dự đoán nhãn cho một văn bản
def predict_naive_bayes(text_tokens, log_priors, log_likelihoods, vocab):
    scores = {}
    for class_label in log_priors:
        score = log_priors[class_label]
        for word in text_tokens:
            if word in vocab:
                score += log_likelihoods[class_label][word]
        scores[class_label] = score
    return max(scores, key=scores.get)

# 7. Tính toán các chỉ số đánh giá chi tiết hơn (Precision, Recall, F1-Score)
def evaluate_metrics(test_set, log_priors, log_likelihoods, vocab):
    #tp: True Positives (Số email spam được dự đoán đúng)
    #tn: True Negatives (Số email ham được dự đoán đúng)
    #fp: False Positives (Số email ham bị dự đoán nhầm thành spam)
    #fn: False Negatives (Số email spam bị dự đoán nhầm thành ham)
    tp = tn = fp = fn = 0

    for tokens, true_label in test_set:
        pred_label = predict_naive_bayes(tokens, log_priors, log_likelihoods, vocab)

        if true_label == 'spam' and pred_label == 'spam':
            tp += 1
        elif true_label == 'ham' and pred_label == 'ham':
            tn += 1
        elif true_label == 'ham' and pred_label == 'spam':
            fp += 1
        elif true_label == 'spam' and pred_label == 'ham':
            fn += 1

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n--- CHI TIẾT ĐÁNH GIÁ (TẬP TEST) ---")
    print(f"Ma trận nhầm lẫn: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"Accuracy  (Độ chính xác tổng): {accuracy*100:.2f}%")
    print(f"Precision (Độ chuẩn xác)     : {precision*100:.2f}%")
    print(f"Recall    (Độ bao phủ)       : {recall*100:.2f}%")
    print(f"F1-Score                     : {f1_score*100:.2f}%")

    return accuracy, precision, recall, f1_score

# 8. Chế độ test thủ công
def interactive_test(log_priors, log_likelihoods, vocab):
    print("\n" + "="*40)
    print("CHẾ ĐỘ TEST THỦ CÔNG (Gõ 'quit' để thoát)")
    print("="*40)

    while True:
        user_input = input("\nNhập nội dung email tiếng Anh: ")

        # Điều kiện thoát vòng lặp
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Đã thoát chế độ test.")
            break

        # Bỏ qua nếu không nhập gì
        if not user_input.strip():
            continue

        # 1. Tiền xử lý giống hệt lúc huấn luyện
        tokens = clean_text(user_input)

        # 2. Đưa vào dự đoán
        prediction = predict_naive_bayes(tokens, log_priors, log_likelihoods, vocab)

        # 3. In kết quả
        if prediction == 'spam':
            print("-> 🔴 KẾT QUẢ: ĐÂY LÀ EMAIL RÁC (SPAM)")
        else:
            print("-> 🟢 KẾT QUẢ: EMAIL BÌNH THƯỜNG (HAM)")

if __name__ == "__main__":
    file_name = 'data_preprocessing/final_huge_spam_dataset.csv'

    dataset = load_data(file_name)
    train_set, test_set = split_data(dataset)
    vocab, word_freq, doc_count = build_frequency_model(train_set)
    log_priors, log_likelihoods = train_naive_bayes(vocab, word_freq, doc_count)
    accuracy, precision, recall, f1_score = evaluate_metrics(test_set, log_priors, log_likelihoods, vocab)
    print(f"Tổng: {len(dataset)} | Train: {len(train_set)} | Test: {len(test_set)}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Spam docs: {doc_count['spam']}, Ham docs: {doc_count['ham']}")
    print(f"Log priors: {log_priors}")

    interactive_test(log_priors, log_likelihoods, vocab)
