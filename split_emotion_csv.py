import csv

# 원본 파일 경로
input_file = 'data/test_ko_emotion.csv'

# 감정별로 저장할 파일 경로
output_files = {
    'depressed': 'data/test_ko_emotion_depressed.csv',
    'surprised': 'data/test_ko_emotion_surprised.csv',
    'angry': 'data/test_ko_emotion_angry.csv',
}

# 감정별 인덱스 (header 기준)
label_indices = {
    'depressed': 2,  # happy, depressed, surprised, angry, neutral
    'surprised': 3,
    'angry': 4,
}

# 각 감정별로 행을 저장할 리스트
emotion_rows = {
    'depressed': [],
    'surprised': [],
    'angry': [],
}

with open(input_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        for emotion, idx in label_indices.items():
            if row[idx] == '1':
                emotion_rows[emotion].append(row)

# header 포함하여 각 감정별로 파일 저장
for emotion, rows in emotion_rows.items():
    with open(output_files[emotion], 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"{emotion} 클래스 샘플 개수: {len(rows)}")

print('분리 완료!') 