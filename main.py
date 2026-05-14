import csv
import json
import os
import re
import random
import math

from underthesea import word_tokenize

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = word_tokenize(text, format="text")

    return text.split()

def load_data(file_path, limit=None):
    dataset = []
    print(f"Đang đọc dữ liệu từ: {file_path}...")

    with open(file_path, 'r', encoding='utf-8-sig') as file:
        reader = csv.reader(file)
        next(reader)

        row_count = 0
        for row in reader:
            if len(row) >= 2:
                raw_label = str(row[0]).strip().lower()

                if raw_label in ['1', 'spam']:
                    label = 'spam'
                elif raw_label in ['0', 'ham']:
                    label = 'ham'
                else:
                    continue

                text = row[1]
                dataset.append((clean_text(text), label))

                row_count += 1
                if limit and row_count >= limit:
                    break

    print(f"Hoàn tất tải {len(dataset)} dòng từ {file_path}.")
    return dataset

def split_data(dataset, split_ratio=0.8):
    random.shuffle(dataset)
    train_size = int(len(dataset) * split_ratio)
    train_set = dataset[:train_size]
    test_set = dataset[train_size:]
    return train_set, test_set

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

# huấn luyện mô hình Naive Bayes
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
            # sử dụng Laplace smoothing (count + 1) để tránh log(0) khi từ không xuất hiện trong lớp đó

    return log_priors, log_likelihoods

def save_model(file_path, vocab, log_priors, log_likelihoods):
    """Lưu 3 biến cốt lõi của mô hình ra file JSON"""
    model_data = {
        'vocab': list(vocab),
        'log_priors': log_priors,
        'log_likelihoods': log_likelihoods
    }
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(model_data, f, ensure_ascii=False, indent=4)
    print(f"💾 Đã lưu mô hình thành công tại: {file_path}")

def load_model(file_path):
    """Đọc file JSON và trả về 3 biến mô hình"""
    with open(file_path, 'r', encoding='utf-8') as f:
        model_data = json.load(f)

    vocab = set(model_data['vocab'])
    log_priors = model_data['log_priors']
    log_likelihoods = model_data['log_likelihoods']

    print(f"📂 Đã tải mô hình thành công từ: {file_path}")
    return vocab, log_priors, log_likelihoods

def predict_naive_bayes(text_tokens, log_priors, log_likelihoods, vocab):
    scores = {}
    for class_label in log_priors:
        score = log_priors[class_label]
        for word in text_tokens:
            if word in vocab:
                score += log_likelihoods[class_label][word]
        scores[class_label] = score
    return max(scores, key=scores.get)

def evaluate_metrics(test_set, log_priors, log_likelihoods, vocab, total_docs, train_size, test_size, doc_count):
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

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)

    print("\n--- CHI TIẾT ĐÁNH GIÁ (TẬP TEST) ---")
    print(f"Ma trận nhầm lẫn: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"Accuracy  (Độ chính xác tổng): {accuracy*100:.2f}%")
    print(f"Precision (Độ chuẩn xác)     : {precision*100:.2f}%")
    print(f"Recall    (Độ bao phủ)       : {recall*100:.2f}%")
    print(f"F1-Score                     : {f1_score*100:.2f}%")
    print(f"Tổng: {total_docs} | Train: {train_size} | Test: {test_size}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Spam docs: {doc_count['spam']}, Ham docs: {doc_count['ham']}")
    print(f"Log priors: {log_priors}")

    return accuracy, precision, recall, f1_score

# 8. Chế độ test thủ công
def interactive_test(log_priors, log_likelihoods, vocab):
    print("\n" + "="*40)
    print("CHẾ ĐỘ TEST THỦ CÔNG (Gõ 'quit' để thoát)")
    print("="*40)

    while True:
        user_input = input("\nNhập nội dung email: ")

        if user_input.lower() == 'quit':
            print("Đã thoát chế độ test.")
            break

        if not user_input.strip():
            continue

        tokens = clean_text(user_input)

        prediction = predict_naive_bayes(tokens, log_priors, log_likelihoods, vocab)

        if prediction == 'spam':
            print("KẾT QUẢ: ĐÂY LÀ EMAIL RÁC (SPAM)")
        else:
            print("KẾT QUẢ: EMAIL BÌNH THƯỜNG (HAM)")

if __name__ == "__main__":
    model_file = "spam_model.json"

    if os.path.exists(model_file):
        print("-> Tìm thấy file mô hình cũ. Đang tải lên...")
        vocab, log_priors, log_likelihoods = load_model(model_file)

    else:
        print("-> Chưa có mô hình. Bắt đầu quá trình đọc dữ liệu và huấn luyện...")
        file_en = 'data_preprocessing/final_huge_spam_dataset.csv'
        dataset_en = load_data(file_en, limit=5000)
        file_vi = 'data_preprocessing/multilingual_huge_dataset.csv'
        dataset_vi = load_data(file_vi, limit=5000)
        combined_dataset = dataset_en + dataset_vi

        train_set, test_set = split_data(combined_dataset)

        vocab, word_freq, doc_count = build_frequency_model(train_set)
        log_priors, log_likelihoods = train_naive_bayes(vocab, word_freq, doc_count)

        evaluate_metrics(test_set, log_priors, log_likelihoods, vocab, len(combined_dataset), len(train_set), len(test_set), doc_count)

        save_model(model_file, vocab, log_priors, log_likelihoods)

    # test tay
    interactive_test(log_priors, log_likelihoods, vocab)
